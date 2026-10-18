import json
import math
import os
import time
from collections import Counter
from pathlib import Path


def validate_full32_gate(records, expected_parameters=18988838144, expected_tensors=4186):
    if len(records) != 32 or {record["actor"]["rank"] for record in records} != set(range(32)):
        raise RuntimeError("Full startup gate requires exactly the actual 32 actor ranks.")
    native = []
    actors = [record["actor"] for record in records]
    for record, actor in zip(records, actors, strict=True):
        if (actor["strategy"], actor["fsdp_size"], actor["sample_dp_size"]) != ("fsdp2", 32, 32):
            raise RuntimeError("Full startup gate found the wrong physical training engine or sample DP.")
        mesh = [rank for row in actor["mesh_ranks"] for rank in (row if isinstance(row, list) else [row])]
        if sorted(mesh) != list(range(32)):
            raise RuntimeError("Full startup gate found the wrong actor mesh.")
        if [actor[key] for key in ("policy_version", "old_version", "export_version")] != [2, 2, 2]:
            raise RuntimeError("Full startup gate requires two real updates and their committed boundary export.")
        if actor["global_parameter_numel"] != expected_parameters or actor["parameter_tensors"] != expected_tensors:
            raise RuntimeError("Full startup gate found the wrong transformer parameter coverage.")
        if actor["optimizer_parameter_tensors"] != expected_tensors:
            raise RuntimeError("Full startup gate found incomplete optimizer coverage.")
        if not 0 < actor["local_parameter_numel"] < expected_parameters // 8 or (
            actor["local_optimizer_state_bytes"] <= 0
        ):
            raise RuntimeError("Full startup gate found replicated parameters or absent Adam ownership.")
        updates = actor["recent_updates"]
        if [update["policy_version"] for update in updates] != [1, 2] or any(
            not math.isfinite(update["global_preclip_norm"]) for update in updates
        ):
            raise RuntimeError("Full startup gate requires both actual finite optimizer steps.")
        replies = record["rollout"]
        if replies is not None:
            if not isinstance(replies, (list, tuple)) or len(replies) != 1 or len(replies[0]) != 2:
                raise RuntimeError("Full startup gate found an unexpected native TP2 topology.")
            native.extend(replies[0])
    if len(native) != 32 or len({(rank["hostname"], rank["physical_gpu"]) for rank in native}) != 32:
        raise RuntimeError("Full startup gate requires all actual 32 native physical GPU receivers.")
    for rank in native:
        state = rank["state"]
        if not rank["ready"] or state["policy_version"] != 2 or state["global_steps"] != 1:
            raise RuntimeError("Full startup gate found an uncommitted native policy.")
        if not state["initial_sync_completed"] or state["source_tensor_count"] != expected_tensors:
            raise RuntimeError("Full startup gate requires both fresh U0 and post-U2 complete synchronization.")
        if state["post_load_checks"] != expected_tensors or not {"packed_q", "packed_k", "packed_v"} <= set(
            state["checked_kinds"]
        ):
            raise RuntimeError("Full startup gate requires independent native packed/TP slice checks.")
    return {"status": "passed_actual_full32_two_update_sharding_and_native_sync_gate", "observed_unix": time.time(),
            "scope": "Fresh real U1/U2 belong to this run, not a warm-start; continuing-rollout acceptance pending.",
            "actors": actors, "native_receivers": native,
            "global_parameter_numel": expected_parameters, "parameter_tensors": expected_tensors,
            "output_change_observed_receivers": sum(any(branch["changed_elements"] > 0 for branch in
                rank["state"].get("fixed_input_output_change", {}).values()) for rank in native
                if rank["state"].get("fixed_input_output_change"))}


def record_full32_startup_gate(records):
    evidence = validate_full32_gate(records)
    output = Path(os.environ["OUTPUT_DIR"]).resolve()
    root = Path(os.environ["OMNI_RUNTIME_ROOT"]).resolve()
    if not output.is_relative_to(root):
        raise ValueError("Full native gate evidence must be attempt-owned.")
    manifest = json.loads((output / "manifest.json").read_text())
    evidence.update(attempt=manifest["attempt"], wandb_run_id=manifest["wandb_run_id"],
                    source_manifest_sha256=manifest["source_manifest_sha256"],
                    source_archive_sha256=manifest["source_archive_sha256"])
    expected_hosts = {node["hostname"]: 8 for node in manifest["nodes"]}
    for category in ("actors", "native_receivers"):
        if dict(Counter(record["hostname"] for record in evidence[category])) != expected_hosts:
            raise RuntimeError("Full native gate actors/receivers do not cover the four authorized hosts exactly.")
    path = output / "full32_native_startup_gate.json"
    temporary = path.with_suffix(".tmp")
    with temporary.open("w") as stream:
        json.dump(evidence, stream, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    return evidence
