# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Isolated A3 preparation and supervision. Never adopts or stops old attempts."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import secrets
import signal
import shlex
import socket
import subprocess
import sys
import time

CAMPAIGN = Path("/m2v_intern/zhenglin/omni_art_ablation_20261016")
CONTROL = Path(__file__).resolve().parent
PORTS = {"10.83.1.23": 21008, "10.83.2.97": 21014, "10.83.2.231": 21014, "10.83.1.141": 21011}
OBJECT_STORE = 4 * 1024**3
TRAIN_SHA = "fb92dc5f97e920c00067ad2129655682166322c7e9897c53c0d75ccdf7c9a7b5"
VAL_SHA = "3c40c064352eb8e2f0c85431eaf47bbe41e0f9c51450b6462afc186dd88d03b7"

def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()

def atomic(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + "." + secrets.token_hex(4) + ".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)

def choose_service_ports():
    for attempt in range(64):
        ray_port = 17000 + secrets.randbelow(2000)
        collective_port = 30000 + secrets.randbelow(2000)
        try:
            with socket.socket() as ray_probe, socket.socket() as collective_probe:
                ray_probe.bind(("0.0.0.0", ray_port))
                collective_probe.bind(("0.0.0.0", collective_port))
        except OSError:
            continue
        return ray_port, collective_port
    raise RuntimeError("No free disjoint Ray and collective service ports found")


def batch_contract(micro):
    if micro not in (1, 2, 4):
        raise ValueError("update microbatch must be 1, 2 or 4")
    return {"prompts": 32, "K": 8, "rollout_samples": 256, "update_samples": 128,
            "updates_per_rollout": 2, "DP": 32, "SP": 1, "fsdp_size": 1,
            "update_microbatch": micro, "gradient_accumulation": 4 // micro,
            "infer_microbatch": 2, "rollout_TP": 2, "rollout_agent_workers": 16,
            "baseline_rollout_agent_workers": 8, "target_updates": None, "target_rollouts": None,
            "run_until_stopped": True}

def allocation():
    rows = [line.split() for line in Path("/etc/mpi/hostfile").read_text().splitlines()
            if line.strip() and not line.lstrip().startswith("#")]
    if len(rows) != 4 or {row[0] for row in rows} != set(PORTS):
        raise RuntimeError("current allocation differs from the declared four hosts; refresh endpoints explicitly")
    return rows

def remote(node, command, timeout=40):
    allocation()
    return subprocess.run(["ssh", "-p", str(node["ssh_port"]), "-o", "BatchMode=yes",
                           "-o", "ConnectTimeout=6", "-o", "StrictHostKeyChecking=yes",
                           "-o", f"UserKnownHostsFile={CAMPAIGN}/extensions/art_joint32/ssh_known_hosts",
                           f"root@{node['ip']}", command], text=True, capture_output=True,
                          timeout=timeout, check=True)

def remote_command(argv, runtime=None):
    env = {} if runtime is None else {
        "OMNI_RUNTIME_ROOT": str(runtime), "OMNI_CONTROL_DIR": str(CONTROL),
        "OMNI_CAMPAIGN_ROOT": str(CAMPAIGN), "PYTHONDONTWRITEBYTECODE": "1",
    }
    return "source /m2v_intern2/zhenglin/my_bashrc.sh\n" + shlex.join(
        ["env", *[f"{key}={value}" for key, value in env.items()], *map(str, argv)])

def validate_audit(audit):
    if time.time() - audit["audit_unix"] > 120:
        raise RuntimeError("node audit expired")
    if len(audit["gpus"]) != 8 or len(audit["compute_apps"]) != 8:
        raise RuntimeError("eight physical GPUs and eight holder rows required")
    if not all("H800" in row for row in audit["gpus"]):
        raise RuntimeError("unexpected GPU type")
    if audit["lease_owner_process"] or audit["lease_live_group_members"]:
        raise RuntimeError("active training lease/processes: investigate incident, never auto-kill")
    if audit.get("lease", {}).get("GPU_LEASE_STATUS") == "quarantined":
        raise RuntimeError("quarantined lease requires explicit incident review")
    if len(audit["processes"]) != 1 or not audit["processes"][0]:
        raise RuntimeError("allocation not exclusively occupied by the expected holder")
    if audit["processes"][0]["cmd"] != "/usr/bin/python /m2v_intern2/zhenglin/gpu_occupy.py":
        raise RuntimeError("unexpected compute process; no cleanup authorized")
    if audit["shared_memory"]["free_bytes"] < OBJECT_STORE + 2 * 1024**3:
        raise RuntimeError("insufficient /dev/shm for Ray object store")

def audit(node):
    result = remote(node, remote_command([
        "timeout", "25", "/usr/bin/python", CAMPAIGN / "scripts/inspect_gpu_node.py"]))
    value = json.loads(result.stdout)
    validate_audit(value)
    return value

def verify_source():
    source = Path(os.environ["SOURCE_CHECKOUT_DIR"]).resolve()
    manifest = Path(os.environ["SOURCE_FILES_MANIFEST"])
    if digest(manifest) != os.environ["SOURCE_FILES_MANIFEST_SHA256"]:
        raise RuntimeError("source manifest digest mismatch")
    files = json.loads(manifest.read_text())
    required = {"scripts/omni_art_joint32/control.py",
                "examples/diffusionart_trainer/ltx2/run_ltx2_t2av_art_joint32.sh",
                "verl_omni/trainer/diffusion/diffusion_art.py"}
    if not required <= set(files):
        raise RuntimeError("source manifest is incomplete")
    for name, expected in files.items():
        path = source / name
        if not path.resolve().is_relative_to(source) or digest(path) != expected:
            raise RuntimeError("source digest mismatch: " + name)

def bounded_io(path, value=None):
    command = [sys.executable, str(CONTROL / "control.py"), "io", str(path)]
    child = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        stdout, stderr = child.communicate(json.dumps(value), timeout=10)
    except subprocess.TimeoutExpired:
        child.kill()
        child.poll()  # Do not wait indefinitely for a process stuck in filesystem D-state.
        raise
    if child.returncode:
        raise subprocess.CalledProcessError(child.returncode, command, stdout, stderr)

def choose_runtime(primary, fallback):
    errors = []
    for root in dict.fromkeys((primary, fallback)):
        try:
            bounded_io(root)
            return root
        except (OSError, subprocess.SubprocessError) as error:
            errors.append(f"{root}: {type(error).__name__}")
            print(errors[-1], file=sys.stderr)
    raise RuntimeError("both write locations unavailable: " + "; ".join(errors))

def pinned(manifest):
    if digest(manifest["launch_manifest"]) != manifest["launch_manifest_sha256"]:
        raise RuntimeError("launch manifest changed")
    for path, expected in json.loads(Path(manifest["launch_manifest"]).read_text()).items():
        if digest(path) != expected:
            raise RuntimeError("pinned runtime changed: " + path)
    for node in manifest["nodes"]:
        if digest(node["env_file"]) != node["env_sha256"]:
            raise RuntimeError("node env changed")
        if json.loads(Path(node["env_file"]).with_suffix(".json").read_text()) != node:
            raise RuntimeError("node descriptor changed")

def prepare(args):
    allocation()
    incident = json.loads(args.incident_review.read_text())
    if incident.get("status") != "reviewed" or not incident.get("evidence_paths") or not incident.get("notes"):
        raise RuntimeError("incident review must contain reviewed status, notes and evidence_paths")
    for path in incident["evidence_paths"]:
        if not Path(path).is_absolute() or not Path(path).is_file():
            raise RuntimeError("incident evidence path missing")
    source = args.source_root.resolve()
    if not source.is_relative_to(CAMPAIGN / "extensions/art_joint32") or CONTROL != source / "scripts/omni_art_joint32":
        raise RuntimeError("deploy the new controls/source under campaign/extensions/art_joint32 first")
    env = dict(os.environ, SOURCE_CHECKOUT_DIR=str(source), SOURCE_FILES_MANIFEST=str(args.source_manifest),
               SOURCE_FILES_MANIFEST_SHA256=args.source_manifest_sha256)
    subprocess.run([sys.executable, str(CONTROL / "control.py"), "verify-source"], env=env, check=True, timeout=120)
    for path, expected in [(CAMPAIGN / "assets/vggsound/parquet/train.parquet", TRAIN_SHA),
                           (CAMPAIGN / "assets/vggsound/parquet/test8.parquet", VAL_SHA)]:
        if digest(path) != expected:
            raise RuntimeError("dataset differs from A0")
    subprocess.run([str(CAMPAIGN / "env/bin/python"), str(CAMPAIGN / "scripts/check_assets.py")],
                   check=True, timeout=60)
    audits = [audit({"ip": ip, "ssh_port": port}) for ip, port in PORTS.items()]
    if len({row["hostname"] for row in audits}) != 4 or socket.gethostname().split(".")[0] != audits[0]["hostname"]:
        raise RuntimeError("prepare/start must run on the allocated head, not developer remote")
    for root in (args.runtime_root, args.fallback_runtime_root):
        if not any(root.is_relative_to(Path(base) / "omni_art_ablation_20261016/extensions/art_joint32")
                   for base in ("/m2v_intern/zhenglin", "/m2v_intern2/zhenglin")):
            raise RuntimeError("write root must be an authorized isolated extension")
    runtime = choose_runtime(args.runtime_root, args.fallback_runtime_root)
    run = "LTX23-A3JOINT32-P32K8-UNBOUNDED-S42-" + time.strftime("%Y%m%dT%H%M%S") + "-" + secrets.token_hex(3)
    output = runtime / "attempts" / run
    for directory in ("logs", "nodes", "provenance"):
        (output / directory).mkdir(parents=True, exist_ok=False)
    (runtime / "tmp").mkdir(exist_ok=True)
    (runtime / "logs").mkdir(exist_ok=True)
    launch_files = list(CONTROL.glob("*.py")) + list(CONTROL.glob("*.sh"))
    launch_files += [CAMPAIGN / "scripts" / name for name in (
        "gpu_lease_lib.sh", "runtime_env.sh", "check_assets.py", "inspect_gpu_node.py")]
    launch_files += [CAMPAIGN / "bin/gpu_free", CAMPAIGN / "bin/gpu_hold"]
    launch_files += [args.source_manifest, args.incident_review, source / "examples/diffusionart_trainer/ltx2/run_ltx2_t2av_art_lora.sh",
                     source / "examples/diffusionart_trainer/ltx2/run_ltx2_t2av_art_joint32.sh"]
    launch = output / "provenance/launch-files.json"
    atomic(launch, {str(path): digest(path) for path in launch_files})
    launch_sha = digest(launch)
    run_id = secrets.token_hex(8)
    ray_port, collective_port = choose_service_ports()
    nodes = []
    source_values = {"SOURCE_CHECKOUT_DIR": str(source), "SOURCE_FILES_MANIFEST": str(args.source_manifest),
                     "SOURCE_FILES_MANIFEST_SHA256": args.source_manifest_sha256,
                     "SOURCE_SNAPSHOT_SHA256": args.source_archive_sha256}
    for rank, ((ip, port), observed) in enumerate(zip(PORTS.items(), audits, strict=True)):
        host = observed["hostname"]
        descriptor = output / "nodes" / f"{host}.json"
        node = {"hostname": host, "ip": ip, "ssh_port": port, "role": "head" if rank == 0 else "worker",
                "attempt": run, "nonce": secrets.token_hex(8), "output": str(output),
                "env_file": str(descriptor.with_suffix(".env")), "planned_holder": observed["processes"][0],
                "launch_manifest": str(launch), "launch_manifest_sha256": launch_sha,
                "tmux_socket": str(runtime / "tmp" / (secrets.token_hex(6) + ".sock")),
                "tmux_session": "a3_" + run_id}
        values = dict(source_values, OMNI_CAMPAIGN_ROOT=str(CAMPAIGN), OMNI_RUNTIME_ROOT=str(runtime),
                      OMNI_CONTROL_DIR=str(CONTROL), OMNI_SHORT_PATH_ROOT=str(runtime.parents[2]),
                      OMNI_OPERATION="training", OMNI_HOLDER_KIND="campaign", OMNI_EXPECTED_HOST=host,
                      OMNI_ATTEMPT=run, OMNI_NONCE=node["nonce"], OMNI_ROLE=node["role"],
                      OMNI_NODE_IP=ip, OMNI_HEAD_IP=next(iter(PORTS)), OMNI_EXPECTED_IPS=",".join(PORTS),
                      OMNI_EXPECTED_PEER_IP=list(PORTS)[1 if rank == 0 else 0],
                      OMNI_HOLDER_PID=observed["processes"][0]["pid"],
                      OMNI_HOLDER_START_TICKS=observed["processes"][0]["start_ticks"],
                      OMNI_RAY_PORT=ray_port, OMNI_RAY_OBJECT_STORE_BYTES=OBJECT_STORE,
                      RAY_ADDRESS=f"{next(iter(PORTS))}:{ray_port}",
                      OMNI_NODE_DESCRIPTOR=str(descriptor), OMNI_LAUNCH_MANIFEST_SHA256=launch_sha,
                      OMNI_VARIANT="A3JOINT32", RUN_NAME=run, OUTPUT_DIR=str(output),
                      WANDB_RUN_ID=run_id, WANDB_RESUME="never", WANDB_MODE="online", WANDB_ENTITY=args.wandb_entity,
                      OMNI_UPDATE_MICROBATCH=args.update_microbatch, OMNI_ART_CACHE_ACCEPTANCE="1")
        for key, directory in {
            "XDG_CONFIG_HOME": "config", "WANDB_CONFIG_DIR": "wandb_config", "WANDB_CACHE_DIR": "wandb_cache",
            "WANDB_DATA_DIR": "wandb_data", "CUPY_CACHE_DIR": "cupy", "NUMBA_CACHE_DIR": "numba",
            "MPLCONFIGDIR": "matplotlib", "TORCH_HOME": "torch",
        }.items():
            values[key] = str(CAMPAIGN / "cache" / directory)
        Path(node["env_file"]).write_text("".join(
            f"export {key}={shlex.quote(str(value))}\n" for key, value in values.items()))
        node["env_sha256"] = digest(node["env_file"])
        atomic(descriptor, node)
        atomic(output / "logs" / f"prepared_{host}.json", observed)
        nodes.append(node)
    manifest = {"attempt": run, "output": str(output), "runtime_root": str(runtime),
                "requested_runtime_root": str(args.runtime_root), "campaign_read_root": str(CAMPAIGN),
                "source_root": str(source), "source_manifest": str(args.source_manifest),
                "source_manifest_sha256": args.source_manifest_sha256, "source_archive_sha256": args.source_archive_sha256,
                "nodes": nodes, "launch_manifest": str(launch), "launch_manifest_sha256": launch_sha,
                "contract": batch_contract(args.update_microbatch), "wandb_run_id": run_id,
                "checkpoint_writes": False, "resume": False, "collective_port": collective_port,
                "gpu_validation": "pending_actual_gate32", "incident_review": str(args.incident_review)}
    atomic(output / "manifest.json", manifest)
    # Resolve the actual new source, not a copied world16 schema certificate.
    config_command = 'source /m2v_intern2/zhenglin/my_bashrc.sh\nsource "$1"\nexport OMNI_CONFIG_ONLY=1 OMNI_CONFIG_OUTPUT="$3"\nexec bash "$2"'
    resolved = subprocess.run(["bash", "-c", config_command, "a3-config", nodes[0]["env_file"],
        str(source / "examples/diffusionart_trainer/ltx2/run_ltx2_t2av_art_joint32.sh"),
        str(output / "provenance/resolved.yaml")],
        check=True, text=True, capture_output=True, timeout=120)
    (output / "logs/config_resolution.log").write_text(resolved.stdout + resolved.stderr)
    subprocess.run([str(CAMPAIGN / "env/bin/python"), str(CONTROL / "validate_config.py"),
                    str(output / "provenance/resolved.yaml"), str(args.update_microbatch)], check=True, timeout=30)
    atomic(output / "prepared.json", {"status": "prepared_cpu_config_only", "manifest_sha256": digest(output / "manifest.json"),
                                     "resolved_sha256": digest(output / "provenance/resolved.yaml")})
    print(json.dumps({"manifest": str(output / "manifest.json"), "actual_write_root": str(runtime),
                      "source_read_root": str(source), "campaign_read_root": str(CAMPAIGN)}, indent=2))

def control(node, action, runtime):
    result = remote(node, remote_command(["timeout", "25", "/usr/bin/python", CONTROL / "node_owner_control.py",
                                          action, str(Path(node["env_file"]).with_suffix(".json"))], runtime))
    return json.loads(result.stdout)

def stop_owned(manifest):
    errors = []
    for node in manifest["nodes"]:
        try:
            control(node, "stop", manifest["runtime_root"])
        except (OSError, ValueError, subprocess.SubprocessError) as error:
            errors.append(f"{node['hostname']}:{type(error).__name__}")
    if errors:
        print("Owned cleanup needs review: " + ",".join(errors), file=sys.stderr)

def supervise(path):
    manifest = json.loads(path.read_text())
    output = Path(manifest["output"])
    lock = (output / "coordinator.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state_file = output / "coordinator_state.json"
    if state_file.exists():
        raise RuntimeError("attempt already started; never re-adopt")
    pinned(manifest)
    prepared = json.loads((output / "prepared.json").read_text())
    if prepared["manifest_sha256"] != digest(path):
        raise RuntimeError("prepared manifest changed")
    allocation()
    # Refresh all four before any lease is acquired. W&B crash is not process death.
    for node in manifest["nodes"]:
        observed = audit(node)
        if observed["hostname"] != node["hostname"] or any(
            observed["processes"][0][key] != node["planned_holder"][key] for key in ("pid", "start_ticks", "cmd")
        ):
            raise RuntimeError("holder identity changed; prepare a fresh attempt after incident review")
        bounded_io(output / "logs" / f"prelaunch_{node['hostname']}.json", observed)
    state = {"status": "starting", "started_unix": time.time(), "attempt": manifest["attempt"],
             "pid": os.getpid(), "scientific_acceptance": "pending_five_finite_updates_and_later_cycle"}
    bounded_io(state_file, state)
    started = False
    stopping = False
    first_terminal = None
    def interrupted(signum, frame):
        raise SystemExit(128 + signum)
    for signum in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT):
        signal.signal(signum, interrupted)
    try:
        for node in manifest["nodes"]:
            started = True
            try:
                remote(node, remote_command(["env", "LD_LIBRARY_PATH=", "tmux", "-S", node["tmux_socket"],
                    "new-session", "-d", "-s", node["tmux_session"], "-c", str(CAMPAIGN),
                    "bash", CONTROL / "launch_training_owner.sh", node["env_file"]], manifest["runtime_root"]))
            except subprocess.TimeoutExpired:
                pass  # Reconcile this exact identity; timeout does not prove absence.
        while True:
            rows = {}
            for node in manifest["nodes"]:
                try:
                    rows[node["hostname"]] = control(node, "status", manifest["runtime_root"])
                except (OSError, ValueError, subprocess.SubprocessError) as error:
                    rows[node["hostname"]] = {"observation_error": type(error).__name__}
            terminals = [row.get("terminal") for row in rows.values()]
            if any(terminals) and first_terminal is None:
                first_terminal = time.time()
            startup_expired = time.time() - state["started_unix"] > 180
            bad = any(t and t["exit_code"] != 0 for t in terminals)
            bad |= first_terminal is not None and time.time() - first_terminal > 180
            bad |= startup_expired and any(
                "observation_error" in row or (not row.get("live") and not row.get("terminal"))
                for row in rows.values())
            stopping |= bad
            state.update(status="stopping_owned_attempt" if stopping else "running_pending_scientific_acceptance",
                         nodes=rows, observed_unix=time.time())
            if all(terminals):
                success = all(row["terminal"]["exit_code"] == 0 and row.get("occupancy_verified") for row in rows.values())
                state["status"] = "finished_pending_scientific_verification" if success else "failed_requires_review"
                bounded_io(state_file, state)
                return 0 if success else 1
            bounded_io(state_file, state)  # Actual I/O only, no capacity floor.
            if stopping:
                stop_owned(manifest)
                if all(not row.get("live") and not row.get("launcher_session_present")
                       and "observation_error" not in row for row in rows.values()):
                    state["status"] = "failed_requires_review"
                    bounded_io(state_file, state)
                    return 1
            time.sleep(15)
    except BaseException:
        if started:
            stop_owned(manifest)
        raise

def start(path):
    manifest = json.loads(path.read_text())
    pinned(manifest)
    allocation()
    observed = audit(manifest["nodes"][0])
    if socket.gethostname().split(".")[0] != observed["hostname"]:
        raise RuntimeError("start only on allocated head")
    runtime = Path(manifest["runtime_root"])
    subprocess.run(["env", "LD_LIBRARY_PATH=", "tmux", "-S", str(runtime / "tmp/coordinator.sock"),
                    "new-session", "-d", "-s", "a3_" + manifest["wandb_run_id"],
                    "bash", "-c", 'source /m2v_intern2/zhenglin/my_bashrc.sh; exec bash "$1" supervise "$2" > "$3" 2>&1',
                    "a3-coordinator", str(CONTROL / "launch.sh"), str(path),
                    str(Path(manifest["output"]) / "logs/coordinator.log")], check=True, timeout=15)

def main():
    command = sys.argv[1]
    if command == "verify-source":
        verify_source()
    elif command == "io":
        path = Path(sys.argv[2])
        value = json.load(sys.stdin)
        if value is None:
            path.mkdir(parents=True, exist_ok=True)
            probe = path / (".io-" + secrets.token_hex(8))
            atomic(probe, {"probe": True})
            probe.unlink()
        else:
            atomic(path, value)
    elif command in ("start", "supervise"):
        path = Path(sys.argv[2]).resolve()
        return start(path) if command == "start" else supervise(path)
    elif command == "prepare":
        parser = argparse.ArgumentParser()
        parser.add_argument("--source-root", type=Path, required=True)
        parser.add_argument("--source-manifest", type=Path, required=True)
        parser.add_argument("--source-manifest-sha256", required=True)
        parser.add_argument("--source-archive-sha256", required=True)
        parser.add_argument("--incident-review", type=Path, required=True)
        parser.add_argument("--wandb-entity", required=True)
        parser.add_argument("--runtime-root", type=Path, default=CAMPAIGN / "extensions/art_joint32")
        parser.add_argument("--fallback-runtime-root", type=Path,
                            default=Path("/m2v_intern2/zhenglin/omni_art_ablation_20261016/extensions/art_joint32"))
        parser.add_argument("--update-microbatch", type=int, choices=(1, 2, 4), default=4)
        prepare(parser.parse_args(sys.argv[2:]))
    else:
        raise ValueError("unsupported action")

if __name__ == "__main__":
    raise SystemExit(main())
