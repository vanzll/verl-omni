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
    nft_flags = [actor.get("policy_kind") == "nft_current_and_frozen_ema" for actor in actors]
    if any(nft_flags) != all(nft_flags):
        raise RuntimeError("Full startup gate cannot mix ART and NFT actor policies.")
    nft = all(nft_flags)
    expected_version = actors[0].get("optimizer_updates_per_rollout") if nft else 2
    if type(expected_version) is not int or expected_version not in (1, 2):
        raise RuntimeError("Full startup gate requires the configured one- or two-update rollout boundary.")
    for record, actor in zip(records, actors, strict=True):
        if (actor["strategy"], actor["fsdp_size"], actor["sample_dp_size"]) != ("fsdp2", 32, 32):
            raise RuntimeError("Full startup gate found the wrong physical training engine or sample DP.")
        mesh = [rank for row in actor["mesh_ranks"] for rank in (row if isinstance(row, list) else [row])]
        if sorted(mesh) != list(range(32)):
            raise RuntimeError("Full startup gate found the wrong actor mesh.")
        if [actor[key] for key in ("policy_version", "old_version", "export_version")] != [expected_version] * 3:
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
        if [update["policy_version"] for update in updates] != list(range(1, expected_version + 1)) or any(
            not math.isfinite(update["global_preclip_norm"]) for update in updates
        ):
            raise RuntimeError("Full startup gate requires both actual finite optimizer steps.")
        if nft and (
            not actor.get("ema_frozen_nonaliasing") or actor.get("base_reference_constructed") is not False
            or actor.get("optimizer_updates_per_rollout") != expected_version
            or type(actor.get("optimizer_updates_per_rollout")) is not int
            or actor.get("ema_updates") != 1 or actor.get("ema_policy_version") != expected_version
            or actor.get("ema_parameter_tensors") != expected_tensors
            or actor.get("ema_global_parameter_numel") != expected_parameters
            or actor.get("ema_local_parameter_numel") != actor["local_parameter_numel"]
        ):
            raise RuntimeError("Full NFT startup gate requires matching frozen EMA shards refreshed once after the configured updates.")
        replies = record["rollout"]
        if replies is not None:
            if not isinstance(replies, (list, tuple)) or len(replies) != 1 or len(replies[0]) != 2:
                raise RuntimeError("Full startup gate found an unexpected native TP2 topology.")
            native.extend(replies[0])
    if len(native) != 32 or len({(rank["hostname"], rank["physical_gpu"]) for rank in native}) != 32:
        raise RuntimeError("Full startup gate requires all actual 32 native physical GPU receivers.")
    for rank in native:
        state = rank["state"]
        if not rank["ready"] or state["policy_version"] != expected_version or state["global_steps"] != 1:
            raise RuntimeError("Full startup gate found an uncommitted native policy.")
        if nft and (state.get("policy_algorithm") != "omni_nft"
                    or state.get("optimizer_updates_per_rollout") != expected_version
                    or type(state.get("optimizer_updates_per_rollout")) is not int):
            raise RuntimeError("Full NFT startup gate found inconsistent native synchronization clock tags.")
        if not state["initial_sync_completed"] or state["source_tensor_count"] != expected_tensors:
            raise RuntimeError("Full startup gate requires both fresh U0 and post-U2 complete synchronization.")
        if state["post_load_checks"] != expected_tensors or not {"packed_q", "packed_k", "packed_v"} <= set(
            state["checked_kinds"]
        ):
            raise RuntimeError("Full startup gate requires independent native packed/TP slice checks.")
    return {"status": f"passed_actual_full32_nft_u{expected_version}_sharding_ema_native_sync_gate" if nft else
            "passed_actual_full32_two_update_sharding_and_native_sync_gate", "observed_unix": time.time(),
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
