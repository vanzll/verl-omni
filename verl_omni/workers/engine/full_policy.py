import json
import os
import socket
import time
from collections import Counter
from contextlib import contextmanager
from pathlib import Path

import torch
import torch.distributed as dist


def local_tensor(value):
    return value.to_local() if hasattr(value, "to_local") else value


def parameter_ownership(module, optimizer):
    parameters = list(module.named_parameters())
    expected = {id(parameter) for _, parameter in parameters if parameter.requires_grad}
    covered = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
    if not expected or expected != covered or len(expected) != len(parameters):
        raise ValueError("Full optimizer must cover every transformer parameter exactly, with no frozen transformer.")
    if sum(len(group["params"]) for group in optimizer.param_groups) != len(covered):
        raise ValueError("Duplicate Full optimizer parameters.")
    dtypes = Counter()
    state_dtypes = Counter()
    parameter_bytes = gradient_bytes = state_bytes = local_numel = global_numel = 0
    shard_specs = []
    for name, parameter in parameters:
        local = local_tensor(parameter)
        global_numel += parameter.numel()
        local_numel += local.numel()
        parameter_bytes += local.numel() * local.element_size()
        dtypes[str(parameter.dtype)] += parameter.numel()
        if parameter.grad is not None:
            gradient = local_tensor(parameter.grad)
            gradient_bytes += gradient.numel() * gradient.element_size()
        placements = [str(placement) for placement in getattr(parameter, "placements", ())]
        shard_spec = {"name": name, "global_shape": list(parameter.shape), "local_shape": list(local.shape),
                      "placements": placements, "dtype": str(parameter.dtype), "requires_grad": parameter.requires_grad,
                      "mesh_ranks": parameter.device_mesh.mesh.tolist() if hasattr(parameter, "device_mesh") else []}
        shard_specs.append(shard_spec)
        for key, value in optimizer.state.get(parameter, {}).items():
            if isinstance(value, torch.Tensor):
                local_state = local_tensor(value)
                state_bytes += local_state.numel() * local_state.element_size()
                state_dtypes[f"{key}:{value.dtype}"] += local_state.numel()
                if key in ("exp_avg", "exp_avg_sq"):
                    shard_spec.setdefault("optimizer_moments", {})[key] = {
                        "global_shape": list(value.shape), "local_shape": list(local_state.shape),
                        "dtype": str(value.dtype),
                        "placements": [str(item) for item in getattr(value, "placements", ())],
                        "mesh_ranks": value.device_mesh.mesh.tolist() if hasattr(value, "device_mesh") else []}
    return {"parameter_tensors": len(parameters), "optimizer_parameter_tensors": len(covered),
            "global_parameter_numel": global_numel, "local_parameter_numel": local_numel,
            "local_parameter_bytes": parameter_bytes, "local_gradient_bytes": gradient_bytes,
            "local_optimizer_state_bytes": state_bytes, "parameter_dtypes": dict(dtypes),
            "optimizer_state_dtypes": dict(state_dtypes), "parameter_shards": shard_specs}


class FullPolicyMixin:
    @property
    def full_finetune(self):
        return bool(getattr(self.model_config, "full_finetune", False))

    def _initialize_full_policy(self):
        self._full_policy_version = 0
        self._full_old_version = None
        self._full_recent_updates = []
        if self.full_finetune:
            if self.model_config.lora_rank != 0 or self.model_config.lora_adapter_path is not None:
                raise ValueError("Full mode cannot create or initialize from PEFT adapters.")
            if self.model_config.algorithm != "diffusion_art":
                raise ValueError("Logical-old Full mode is only implemented for cached DiffusionART.")

    def _full_event(self, kind, **values):
        output = os.environ.get("OUTPUT_DIR")
        if output is None:
            return
        root = Path(os.environ["OMNI_RUNTIME_ROOT"]).resolve()
        directory = Path(output).resolve()
        if not directory.is_relative_to(root):
            raise ValueError("Full audit output must be attempt-owned.")
        rank = dist.get_rank() if dist.is_initialized() else 0
        record = {"event": kind, "rank": rank, "observed_unix": time.time(),
                  "policy_version": self._full_policy_version, "old_version": self._full_old_version, **values}
        with (directory / "logs" / f"full32_rank_{rank}.jsonl").open("a") as stream:
            stream.write(json.dumps(record) + "\n")

    def copy_adapter(self, source="default", target="old"):
        if not self.full_finetune:
            return super().copy_adapter(source=source, target=target)
        if (source, target) != ("default", "old") or self._full_policy_version % 2:
            raise ValueError("Full logical-old snapshots are only valid at complete two-update boundaries.")
        self._full_old_version = self._full_policy_version
        self._full_event("old_boundary_snapshot")

    def ema_update_adapter(self, source="default", target="old", decay=0.0):
        if not self.full_finetune:
            return super().ema_update_adapter(source=source, target=target, decay=decay)
        if decay != 0.0:
            raise ValueError("Full cached ART requires hard boundary snapshots, not EMA.")
        return self.copy_adapter(source=source, target=target)

    @contextmanager
    def use_adapter(self, name):
        if not self.full_finetune:
            with super().use_adapter(name):
                yield
            return
        if name != "default":
            raise ValueError("Full logical-old is an export boundary, never an additional old/reference forward.")
        yield

    def _full_export_version(self, adapter_name):
        if adapter_name != "old":
            raise ValueError("Full ART export requires explicit logical-old, not an ignored named adapter.")
        if self._full_old_version != self._full_policy_version or self._full_policy_version % 2:
            raise ValueError("Full export is forbidden before the two-update policy boundary is committed.")
        return self._full_policy_version

    def _full_parameter_samples(self):
        names = []
        samples = []
        for name, parameter in self.module.named_parameters():
            local = local_tensor(parameter.detach()).reshape(-1)
            if local.numel():
                names.append(name)
                samples.append(local[local.numel() // 2].float())
        return names, torch.stack(samples).detach().cpu()

    def _full_memory(self):
        device = next(self.module.parameters()).device
        if device.type != "cuda":
            return {}
        return {"cuda_allocated_bytes": torch.cuda.memory_allocated(device),
                "cuda_reserved_bytes": torch.cuda.memory_reserved(device),
                "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(device)}

    def full_policy_state(self):
        if not self.full_finetune:
            raise ValueError("Full audit is not an adapter-policy audit.")
        self._assert_full_runtime_sharding(optimizer_initialized=True)
        ownership = parameter_ownership(self.module, self.optimizer)
        ownership.pop("parameter_shards")
        return {"rank": dist.get_rank(), "hostname": socket.gethostname().split(".")[0],
                "policy_version": self._full_policy_version, "old_version": self._full_old_version,
                "export_version": getattr(self, "_full_last_export_version", None),
                "strategy": self.engine_config.strategy, "fsdp_size": self.engine_config.fsdp_size,
                "sample_dp_size": self.get_data_parallel_size(), "mesh_ranks": self.device_mesh.mesh.tolist(),
                "recent_updates": self._full_recent_updates,
                **ownership, **self._full_memory()}

    def _assert_full_runtime_sharding(self, *, optimizer_initialized=False):
        from torch.distributed.tensor import DTensor

        if self.engine_config.strategy != "fsdp2" or self.engine_config.fsdp_size != 32:
            raise RuntimeError("Full ART requires the actual FSDP2 engine with 32-rank sharding.")
        if self.get_data_parallel_size() != 32 or sorted(self.device_mesh.mesh.reshape(-1).tolist()) != list(range(32)):
            raise RuntimeError("Full ART must retain all 32 sample-DP ranks in the FSDP mesh.")
        if not self.engine_config.reshard_after_forward:
            raise RuntimeError("Full ART cannot retain full parameters after forward.")
        initialized_moments = 0
        for name, parameter in self.module.named_parameters():
            if not isinstance(parameter, DTensor) or not any(
                placement.is_shard() for placement in parameter.placements
            ):
                raise RuntimeError(f"Full parameter is not physically FSDP-sharded: {name}")
            if sorted(parameter.device_mesh.mesh.reshape(-1).tolist()) != list(range(32)):
                raise RuntimeError(f"Full parameter belongs to an incomplete FSDP mesh: {name}")
            if parameter.to_local().numel() >= parameter.numel() and parameter.numel() >= 32:
                raise RuntimeError(f"Full parameter still has replicated local storage: {name}")
            for key in ("exp_avg", "exp_avg_sq"):
                moment = self.optimizer.state.get(parameter, {}).get(key)
                if optimizer_initialized and parameter.grad is not None and moment is None:
                    raise RuntimeError(f"Full Adam did not initialize a moment for an active parameter: {name}/{key}")
                if moment is not None:
                    if not isinstance(moment, DTensor) or moment.placements != parameter.placements:
                        raise RuntimeError(f"Full Adam moment is not owned like its parameter: {name}/{key}")
                    if not torch.equal(moment.device_mesh.mesh, parameter.device_mesh.mesh):
                        raise RuntimeError(f"Full Adam moment mesh differs from its parameter: {name}/{key}")
                    if moment.shape != parameter.shape or moment.to_local().shape != parameter.to_local().shape:
                        raise RuntimeError(f"Full Adam local/global shapes differ from its parameter: {name}/{key}")
                    initialized_moments += 1
        if optimizer_initialized and initialized_moments == 0:
            raise RuntimeError("Full optimizer returned without initializing any sharded Adam moments.")

    def _validate_full_cached_window(self, data, timesteps_key):
        versions = data.get("art_policy_version")
        windows = data.get("art_update_window")
        if versions is None or windows is None or self._full_old_version is None:
            raise ValueError("Full training requires native cache versions and explicit two-window identities.")
        window = self._full_policy_version - self._full_old_version
        if window not in (0, 1) or not bool((versions == self._full_old_version).all()):
            raise ValueError("Full cached old target does not match the frozen rollout boundary.")
        counts = data.get("art_train_step_count")
        if counts is not None and (counts.shape != versions.shape or counts.dtype != torch.int64):
            raise ValueError("Full cached ART requires an int64 timestep count per trajectory.")
        expected_steps = 19 if counts is None else int(counts.reshape(-1)[0])
        if counts is not None and not bool((counts == expected_steps).all()):
            raise ValueError("Full cached ART requires one consistent configured timestep count.")
        if (
            not bool((windows == window).all())
            or expected_steps not in (19, 20)
            or data[timesteps_key].shape[1] != expected_steps
        ):
            raise ValueError("Full cached ART requires the expected window and all configured 19 or 20 timesteps.")
        if counts is not None:
            rollout_times = data.get("all_timesteps")
            if rollout_times is None or rollout_times.shape != (len(versions), 20):
                raise ValueError("Full cached ART requires the original 20-position rollout time grid.")
            expected_times = rollout_times[:, :expected_steps].sort(dim=1).values
            if not torch.equal(data[timesteps_key].sort(dim=1).values, expected_times):
                raise ValueError("Full cached ART training positions do not match the configured rollout time grid.")
        self._full_event("cached_window", update_window=window, cache_version=int(versions.reshape(-1)[0]),
                         samples=len(versions), training_timesteps=expected_steps)

    def _full_step_applied(self, before, grad_norm):
        names, previous = before
        after_names, current = self._full_parameter_samples()
        if names != after_names:
            raise RuntimeError("Full parameter ownership changed across optimizer step.")
        changed = previous != current
        changed_count = torch.tensor(
            int(changed.sum()), device=next(self.module.parameters()).device, dtype=torch.int64
        )
        if dist.is_initialized():
            dist.all_reduce(changed_count)
        self._full_policy_version += 1
        if hasattr(self, "engine_config"):
            self._assert_full_runtime_sharding(optimizer_initialized=True)
        changes = [{"name": names[index], "before": previous[index].item(), "after": current[index].item()}
                   for index in torch.where(changed)[0].tolist()[:12]]
        ownership = parameter_ownership(self.module, self.optimizer)
        if self._full_policy_version != 1:
            ownership.pop("parameter_shards")
        self._full_last_step = {"successful_optimizer_updates": self._full_policy_version,
                                "global_preclip_norm": float(grad_norm),
                                "clip_coefficient": min(
                                    1.0, self.optimizer_config.clip_grad / (float(grad_norm) + 1e-6)
                                ),
                                "local_changed_parameter_samples": int(changed.sum()),
                                "global_changed_parameter_samples": int(changed_count.item()),
                                **ownership, **self._full_memory()}
        self._full_recent_updates = (self._full_recent_updates + [{
            "policy_version": self._full_policy_version, "global_preclip_norm": float(grad_norm),
            "global_changed_parameter_samples": int(changed_count.item())}])[-2:]
        self._full_event("optimizer_step", changed_examples=changes, **self._full_last_step)
