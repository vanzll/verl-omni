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

"""CPU-only A3 argv, topology, ownership and immutable baseline regressions."""
import contextlib
import ast
import itertools
import importlib.util
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[2]
SCRIPTS = REPO / "scripts/omni_art_joint32"
LEGACY = Path(os.environ.get("OMNI_LEGACY_AUDIT", str(REPO.parent / "paper_data/omni-art-ablation-20261016/runtime-audit-20261017")))
OLD_HASHES = {
    "scripts/gpu_lease_lib.sh": "c6ad96fb04fc5bf1fcaafdfa813f2246f3b9c1828de71ad370c8be3ee98245c7",
    "bin/gpu_hold": "afebcab73186e6c233d5fd00d11449a4ec672d98ecdd2aff7d42cd72d03ac8de",
    "bin/gpu_free": "afebcab73186e6c233d5fd00d11449a4ec672d98ecdd2aff7d42cd72d03ac8de",
    "scripts/run_node_owner.sh": "388fe4d9d4a9df7b35df08fbb71698d4a5a03a650a1d008f3196a5622d14fcb5",
    "scripts/node_owner_control.py": "7c0a964508ad3abca24c014ef4ea6f36ac0cfa1efa59e4af9b2286332fe2fe32",
    "scripts/node_payload.sh": "5b4ceb38289ca38dc77e295fc06f2e7dbc5a0b0035b4c2f57a6ef365de7b5643",
    "scripts/run_art.sh": "fadb58fe0dc252182974700d050036bdf35a0a046e7b598b228049745e1bd171",
    "scripts/coordinate_pair.py": "954a7d28d29d351a8a355dd0d7af5c8233390173fc622442755f9d44f01cb6ba",
    "scripts/prepare_pair_attempt.py": "412a7ad625d01dc1a9fb5cd9bc5b7d86e7974445346273b26219287e7c1b0fae",
    "scripts/runtime_env.sh": "ca654bffd3926e176235d02087ce72ce688b782b25bc250c4f0e4041fdefa0a9"
}

def load(name):
    spec = importlib.util.spec_from_file_location("a3_" + name, SCRIPTS / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

control = load("control")
gate = load("gate32")
validator = load("validate_config")

class LaunchTests(unittest.TestCase):
    def test_service_ports_never_overlap_ray_worker_range(self):
        bound = []

        class Probe:
            def __enter__(self):
                return self

            def __exit__(self, *arguments):
                return None

            def bind(self, address):
                bound.append(address[1])

        with patch.object(control.secrets, "randbelow", side_effect=[1999, 1999]), \
                patch.object(control.socket, "socket", side_effect=lambda: Probe()):
            ray_port, collective_port = control.choose_service_ports()
        self.assertEqual((ray_port, collective_port), (18999, 31999))
        self.assertEqual(bound, [ray_port, collective_port])
        self.assertTrue(all(not 20000 <= port <= 29999 for port in bound))

    def test_busy_service_port_is_rejected_before_prepare_returns(self):
        bound = []

        class Probe:
            def __enter__(self):
                return self

            def __exit__(self, *arguments):
                return None

            def bind(self, address):
                bound.append(address[1])
                if len(bound) == 2:
                    raise OSError("port already occupied")

        with patch.object(control.secrets, "randbelow", side_effect=[1, 1, 2, 2]), \
                patch.object(control.socket, "socket", side_effect=lambda: Probe()):
            self.assertEqual(control.choose_service_ports(), (17002, 30002))
        self.assertEqual(bound, [17001, 30001, 17002, 30002])

    def test_native_zmq_uses_short_owned_runtime_path(self):
        import uuid

        import zmq

        with tempfile.TemporaryDirectory() as directory:
            runtime = Path(directory) / ("long-runtime-" + "nested-" * 15)
            short_root = Path(os.environ.get("OMNI_TEST_IPC_SHORT_ROOT", directory))
            if len(str(short_root)) + 50 > 107:
                self.skipTest("set OMNI_TEST_IPC_SHORT_ROOT to an authorized short directory")
            attempt = "ipc-test-" + uuid.uuid4().hex
            host = "test-host"
            environment = dict(os.environ, OMNI_RUNTIME_ROOT=str(runtime),
                               OMNI_ATTEMPT=attempt, OMNI_EXPECTED_HOST=host,
                               OMNI_NONCE=uuid.uuid4().hex, OMNI_SHORT_PATH_ROOT=str(short_root))
            command = 'source "$1"\nprintf "%s\\n" "$RAY_TMPDIR" "$TMPDIR" "$VLLM_RPC_BASE_PATH"'
            result = subprocess.check_output(
                ["bash", "-c", command, "ipc-test", str(SCRIPTS / "ray_temp_env.sh")],
                env=environment, text=True, timeout=10,
            ).splitlines()
            alias = Path(result[0])
            try:
                self.assertEqual(result, [str(alias)] * 3)
                self.assertEqual(alias.resolve(), runtime / "tmp/ray" / attempt / host)
                socket_path = alias / str(uuid.uuid4())
                self.assertLessEqual(len(str(socket_path).encode()), 107)
                context = zmq.Context()
                channel = context.socket(zmq.PAIR)
                try:
                    channel.bind("ipc://" + str(socket_path))
                    self.assertTrue(socket_path.exists())
                finally:
                    channel.close(linger=0)
                    context.term()
            finally:
                alias.unlink()

    def test_unbounded_training_and_finite_default(self):
        tree = ast.parse((REPO / "verl_omni/trainer/diffusion/ray_diffusion_trainer.py").read_text())
        base = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "BaseRayDiffusionTrainer")
        methods = [n for n in base.body if isinstance(n, ast.FunctionDef) and n.name in ("_training_epochs", "_is_last_training_step")]
        namespace = {"count": itertools.count}
        exec(compile(ast.Module(body=methods, type_ignores=[]), "trainer-budget", "exec"), namespace)
        class Config(dict):
            __getattr__ = dict.__getitem__
        trainer = SimpleNamespace(config=SimpleNamespace(trainer=Config(run_until_stopped=True, total_epochs=1)),
                                  total_training_steps=None, global_steps=1)
        self.assertEqual(list(itertools.islice(namespace["_training_epochs"](trainer, 0), 4)), [0, 1, 2, 3])
        for step in (75, 150, 1000000):
            trainer.global_steps = step
            self.assertFalse(namespace["_is_last_training_step"](trainer))
        trainer.config.trainer = Config(total_epochs=3)
        trainer.total_training_steps = 75
        self.assertEqual(list(namespace["_training_epochs"](trainer, 1)), [1, 2])
        trainer.global_steps = 74
        self.assertFalse(namespace["_is_last_training_step"](trainer))
        trainer.global_steps = 75
        self.assertTrue(namespace["_is_last_training_step"](trainer))
        setup = next(n for n in base.body if isinstance(n, ast.FunctionDef) and n.name == "_create_dataloader")
        guard = next(n for n in setup.body if isinstance(n, ast.If) and "run_until_stopped" in ast.unparse(n.test))
        code = compile(ast.Module(body=[guard], type_ignores=[]), "budget-guard", "exec")
        for scheduler, ratio, valid in (("constant", 0, True), ("cosine", 0, False), ("constant", 0.1, False)):
            trainer.config.trainer = Config(run_until_stopped=True)
            trainer.config.actor_rollout_ref = SimpleNamespace(actor=SimpleNamespace(optim=Config(lr_scheduler_type=scheduler, lr_warmup_steps_ratio=ratio)))
            if valid:
                exec(code, {"self": trainer, "total_training_steps": 75})
                self.assertIsNone(trainer.total_training_steps)
            else:
                with self.assertRaises(ValueError):
                    exec(code, {"self": trainer, "total_training_steps": 75})

    def test_batch_boundaries(self):
        for micro in (1, 2, 4):
            contract = control.batch_contract(micro)
            self.assertEqual(32 * micro * contract["gradient_accumulation"], 128)
            self.assertIsNone(contract["target_rollouts"])
            self.assertIsNone(contract["target_updates"])
            self.assertTrue(contract["run_until_stopped"])
            self.assertEqual(contract["infer_microbatch"], 2)
        with self.assertRaises(ValueError):
            control.batch_contract(8)

    def test_readiness_exact_four(self):
        nodes = [{"Alive": True, "NodeManagerAddress": ip, "Resources": {"GPU": 8}} for ip in control.PORTS]
        self.assertEqual(len(gate.validate_nodes(nodes, control.PORTS)), 4)
        for bad in (nodes[:2], nodes + nodes[:1], [dict(nodes[0], Resources={"GPU": 7})] + nodes[1:]):
            with self.assertRaises(RuntimeError):
                gate.validate_nodes(bad, control.PORTS)

    def test_gate_rank_zero_is_manifest_head(self):
        nodes = [{"NodeManagerAddress": ip, "NodeID": ip} for ip in reversed(control.PORTS)]
        ordered = gate.ordered_nodes(nodes, [{"ip": ip} for ip in control.PORTS])
        self.assertEqual(ordered[0]["NodeManagerAddress"], next(iter(control.PORTS)))

    def test_active_lease_rejected_without_remote_actions(self):
        audit = {"audit_unix": time.time(), "gpus": ["H800"] * 8, "compute_apps": ["1"] * 8,
                 "lease_owner_process": {"pid": 42}, "lease_live_group_members": [],
                 "processes": [], "shared_memory": {"free_bytes": 8 * 1024**3}}
        with self.assertRaisesRegex(RuntimeError, "active training lease"):
            control.validate_audit(audit)

    def test_storage_fallback_bounded_and_explicit(self):
        primary, fallback = Path("/primary"), Path("/fallback")
        with patch.object(control, "bounded_io", side_effect=[subprocess.TimeoutExpired("io", 10), None]) as probe:
            self.assertEqual(control.choose_runtime(primary, fallback), fallback)
            self.assertEqual(probe.call_count, 2)
        with patch.object(control, "bounded_io", side_effect=OSError("ENOSPC")):
            with self.assertRaisesRegex(RuntimeError, "both write locations"):
                control.choose_runtime(primary, fallback)

    def test_legacy_unchanged_and_cleanup_preserved(self):
        if not LEGACY.exists():
            self.skipTest("set OMNI_LEGACY_AUDIT to the read-only baseline")
        for name, expected in OLD_HASHES.items():
            self.assertEqual(control.digest(LEGACY / name), expected, name)
        old = (LEGACY / "scripts/run_node_owner.sh").read_text()
        new = (SCRIPTS / "run_node_owner.sh").read_text()
        before = old[old.index("cleanup_done=0"):old.index("holder_is_exact\nassert_only", old.index("cleanup_done=0"))]
        after = new[new.index("cleanup_done=0"):new.index("holder_is_exact\nassert_only", new.index("cleanup_done=0"))]
        self.assertEqual(before.replace("$campaign/scripts/record_gpu_reservation.py", "$control/record_gpu_reservation.py"), after)
        self.assertIn('source "$campaign/scripts/gpu_lease_lib.sh"', new)
        self.assertNotIn('trap "gpu_hold"', new)

    def test_recipe_final_argv_and_failure_status(self):
        for micro in (1, 2, 4):
            with self.subTest(micro=micro), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                (root / "bin").mkdir()
                shim = root / "bash_env"
                shim.write_text('source() { case "$1" in */my_bashrc.sh|*/runtime_env.sh|*/use_source_snapshot.sh) return 0;; *) builtin source "$@";; esac; }\n')
                python = root / "bin/python3"
                python.write_text("#!" + sys.executable + "\nimport json,os,sys\nopen(os.environ['CAPTURE'],'w').write(json.dumps(sys.argv[1:]))\nsys.exit(17)\n")
                python.chmod(0o755)
                capture = root / "argv.json"
                env = dict(os.environ, BASH_ENV=str(shim), CAPTURE=str(capture),
                           PATH=str(root / "bin") + ":" + os.environ["PATH"], OMNI_CAMPAIGN_ROOT=str(root),
                           OMNI_CONTROL_DIR=str(SCRIPTS), OMNI_RUNTIME_ROOT=str(root),
                           SOURCE_CHECKOUT_DIR=str(REPO), SOURCE_SNAPSHOT_SHA256="a" * 64,
                           SOURCE_FILES_MANIFEST_SHA256="b" * 64, OUTPUT_DIR=str(root / "output"),
                           RUN_NAME="A12JOINT32-cpu", WANDB_RUN_ID="fresh", WANDB_RESUME="never",
                           RAY_ADDRESS="head:17000", PYTHONPATH=str(REPO), OMNI_UPDATE_MICROBATCH=str(micro))
                result = subprocess.run(["bash", str(REPO / "examples/diffusionart_trainer/ltx2/run_ltx2_t2av_art_joint32.sh")],
                                        env=env, capture_output=True, text=True, timeout=15)
                self.assertEqual(result.returncode, 17, result.stderr)
                argv = json.loads(capture.read_text())
                self.assertEqual(argv[:2], ["-m", "verl_omni.trainer.main_diffusion"])
                values = dict(arg.lstrip("+").split("=", 1) for arg in argv[2:])
                for key, expected in validator.expected(micro).items():
                    if key.endswith(".foreach"):
                        self.assertEqual(values[key.rsplit(".", 1)[0]], "{foreach: false}")
                        continue
                    actual = values[key]
                    if isinstance(expected, bool):
                        self.assertEqual(actual.lower(), str(expected).lower(), key)
                    elif isinstance(expected, (float, int)):
                        self.assertEqual(float(actual), expected, key)
                    elif expected is None:
                        self.assertEqual(actual, "null", key)
                    elif isinstance(expected, dict):
                        parsed = dict(item.split(":") for item in actual.strip("{}").split(","))
                        self.assertEqual({key.strip(): float(value) for key, value in parsed.items()}, expected)
                    elif isinstance(expected, list):
                        from omegaconf import OmegaConf
                        parsed = OmegaConf.to_container(OmegaConf.from_dotlist(["value=" + actual]))
                        self.assertEqual(parsed["value"], expected, key)
                    else:
                        self.assertEqual(actual, expected, key)
                self.assertEqual(values["reward.reward_functions.hpsv3.routing_weights"], "{video:1.5,audio:0.0}")
                self.assertEqual(values["reward.reward_functions.audiobox.routing_weights"], "{video:0.0,audio:0.5}")

    def test_ownership_nonce_ticks_command_and_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            (root / "logs").mkdir()
            env_file = root / "node.env"
            env_file.write_text("# fixture")
            launch = root / "launch.json"
            launch.write_text("{}")
            node = {"hostname": __import__("socket").gethostname().split(".")[0], "output": str(root),
                    "attempt": "cpu", "nonce": "nonce", "env_file": str(env_file),
                    "env_sha256": control.digest(env_file), "launch_manifest": str(launch),
                    "launch_manifest_sha256": control.digest(launch)}
            descriptor = root / "node.json"
            descriptor.write_text(json.dumps(node))
            with patch.dict(os.environ, OMNI_RUNTIME_ROOT=str(root), OMNI_CAMPAIGN_ROOT=str(root)):
                owner = load("node_owner_control")
            current = {"pid": 42, "ticks": "123", "state": "S",
                       "argv": ["bash", str(SCRIPTS / "run_node_owner.sh"), str(env_file)]}
            def call(action, *args):
                output = io.StringIO()
                with patch.object(sys, "argv", ["owner", action, str(descriptor), *map(str, args)]), contextlib.redirect_stdout(output):
                    owner.main()
                return json.loads(output.getvalue()) if output.getvalue() else None
            with patch.object(owner, "process", return_value=current):
                call("identity", 42)
                call("verify-self", 42)
                identity_file = root / "logs" / f"owner_identity_{node['hostname']}.json"
                identity = json.loads(identity_file.read_text())
                identity_file.write_text(json.dumps(dict(identity, nonce="other")))
                with self.assertRaises(AssertionError):
                    call("stop")
                identity_file.write_text(json.dumps(dict(identity, start_ticks="124")))
                self.assertFalse(call("stop")["signal_sent"])
                identity_file.write_text(json.dumps(identity))
                with patch.object(owner, "process", return_value=dict(current, argv=["unrelated"])):
                    with self.assertRaises(AssertionError):
                        call("stop")
                launch.write_text('{"tampered":true}')
                with self.assertRaises(AssertionError):
                    call("verify-self", 42)
                launch.write_text("{}")
                with patch.object(os, "pidfd_open", return_value=99, create=True), patch.object(os, "close"), patch.object(signal, "pidfd_send_signal", create=True) as send:
                    self.assertTrue(call("stop")["signal_sent"])
                    send.assert_called_once_with(99, signal.SIGTERM)

    def test_shell_syntax(self):
        for file in [*SCRIPTS.glob("*.sh"), REPO / "examples/diffusionart_trainer/ltx2/run_ltx2_t2av_art_joint32.sh"]:
            subprocess.run(["bash", "-n", str(file)], check=True)

    def test_config_save_ignores_noisy_stdout(self):
        resolver = load("resolve_config")
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "resolved.yaml"
            def compose(**kwargs):
                print("registration noise and tee logs are not YAML")
                return {"trainer": {"nnodes": 4}}
            omega = SimpleNamespace(resolve=lambda config: None,
                save=lambda config, path, resolve: Path(path).write_text(json.dumps(config)))
            hydra = SimpleNamespace(compose=compose, initialize_config_dir=lambda **kwargs: contextlib.nullcontext())
            stdout = io.StringIO()
            with patch.dict(sys.modules, hydra=hydra, omegaconf=SimpleNamespace(OmegaConf=omega)), \
                 patch.dict(os.environ, SOURCE_CHECKOUT_DIR=str(REPO), OMNI_CONFIG_OUTPUT=str(destination)), \
                 patch.object(sys, "argv", ["resolve", "-m", "verl_omni.trainer.main_diffusion"]), \
                 contextlib.redirect_stdout(stdout):
                resolver.main()
            self.assertIn("registration noise", stdout.getvalue())
            self.assertEqual(json.loads(destination.read_text()), {"trainer": {"nnodes": 4}})

if __name__ == "__main__":
    unittest.main()
