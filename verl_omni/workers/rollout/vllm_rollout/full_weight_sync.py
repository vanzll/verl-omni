from math import prod

import torch

from verl_omni.utils.tensor_audit import tensor_sample


def unpack_full_policy_sync_replies(replies, world_size):
    if not isinstance(replies, (list, tuple)) or len(replies) != 1:
        raise RuntimeError("Full ART requires exactly one native diffusion replica per server.")
    executor_replies = replies[0]
    if not isinstance(executor_replies, (list, tuple)) or len(executor_replies) != 1:
        raise RuntimeError("Full ART requires one aggregated native executor reply.")
    ranks = executor_replies[0]
    if not isinstance(ranks, (list, tuple)) or len(ranks) != world_size:
        raise RuntimeError("Full policy commit requires every native rollout rank acknowledgement.")
    if any(not isinstance(rank, dict) for rank in ranks):
        raise RuntimeError("Full native rollout returned a malformed rank acknowledgement.")
    if {rank.get("native_rank") for rank in ranks} != set(range(world_size)) or any(
        rank.get("native_world_size") != world_size for rank in ranks
    ):
        raise RuntimeError("Full native rollout returned incomplete or duplicate rank membership.")
    if any(not rank.get("hostname") or not rank.get("physical_gpu") for rank in ranks) or len({
        (rank["hostname"], rank["physical_gpu"]) for rank in ranks
    }) != world_size:
        raise RuntimeError("Full native rollout rank acknowledgements do not cover distinct physical GPUs.")
    return list(ranks)


def fixed_ltx2_output_probe(model):
    from vllm_omni.diffusion.models.ltx2.ltx2_guidance import LTXGuidanceExecutor

    device = model.proj_in.weight.device
    dtype = model.proj_in.weight.dtype
    config = model.config

    def fixed(shape, start, end):
        return torch.linspace(
            start, end, steps=prod(shape), device=device, dtype=torch.float32
        ).reshape(shape).to(dtype)

    video_channels = config.caption_channels if hasattr(model, "caption_projection") else config.cross_attention_dim
    audio_channels = (
        config.caption_channels if hasattr(model, "audio_caption_projection") else config.audio_cross_attention_dim
    )
    devices = [device] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices), torch.inference_mode():
        outputs = model(hidden_states=fixed((1, 4, config.in_channels), -0.5, 0.5),
                        audio_hidden_states=fixed((1, 4, config.audio_in_channels), 0.4, -0.4),
                        encoder_hidden_states=fixed((1, 2, video_channels), -0.2, 0.3),
                        audio_encoder_hidden_states=fixed((1, 2, audio_channels), 0.1, -0.25),
                        **LTXGuidanceExecutor.timestep_kwargs(
                            torch.full((1,), 800.0, device=device, dtype=torch.float32), 4, 4),
                        num_frames=1, height=2, width=2, fps=24.0, audio_num_frames=4, return_dict=False)
        result = {}
        for branch, tensor in zip(("video", "audio"), outputs, strict=True):
            values = tensor.detach().float().cpu()
            if not bool(torch.isfinite(values).all()):
                raise RuntimeError(f"Nonfinite Full native fixed-input {branch} output.")
            result[branch] = {
                "shape": list(tensor.shape), "dtype": str(tensor.dtype), "values": values.reshape(-1).tolist()
            }
        return result


class FullWeightCoverage:
    def __init__(self, model, metadata):
        self.model = model
        self.metadata = metadata
        self.parameters = dict(model.named_parameters())
        self.source = {}
        self.loaded = set()
        self.parts = {}
        self.samples = {}
        self.post_load_checks = 0
        self.bucket_count = 0
        self.maximum_bucket_bytes = 0
        self.post_load_samples = {}
        self.checked_kinds = set()

    def _target(self, source):
        name = source.removeprefix("transformer.")
        if name == source:
            raise ValueError(f"Full ART may only load transformer weights: {source}")
        remap = getattr(self.model, "_phase_adapter_parameter_name", lambda value: value)
        for parameter_name, source_name, shard_id in self.model.stacked_params_mapping:
            if source_name in name:
                target = remap(name.replace(source_name, parameter_name))
                if target in self.parameters:
                    return target, shard_id
        target = remap(name)
        if target not in self.parameters:
            raise ValueError(f"Unknown Full transformer source key, not silently skipped: {source}")
        return target, None

    def _local_slice(self, target, part, tensor):
        parameter = self.parameters[target]
        module = self.model.get_submodule(target.rsplit(".", 1)[0]) if "." in target else self.model
        tp_size = getattr(module, "tp_size", self.metadata.get("tensor_parallel_size", 1))
        tp_rank = getattr(module, "tp_rank", self.metadata.get("tensor_parallel_rank", 0))
        if part is not None:
            offset = module._get_shard_offset_mapping(part)
            size = module._get_shard_size_mapping(part)
            dimension = getattr(parameter, "output_dim", 0)
            rank = tp_rank if part == "q" else tp_rank // getattr(module, "num_kv_head_replicas", 1)
            expected = tensor.narrow(dimension, rank * size, size)
            actual = parameter.narrow(dimension, offset, size)
            kind = f"packed_{part}"
        elif parameter.shape == tensor.shape:
            expected, actual, kind = tensor, parameter, "replicated"
        else:
            dimensions = [index for index, (source, local) in enumerate(zip(tensor.shape, parameter.shape, strict=True))
                          if source != local]
            if len(dimensions) != 1 or tp_size <= 1:
                raise ValueError(f"Unsupported Full TP shape mapping for {target}")
            dimension = dimensions[0]
            size = parameter.shape[dimension]
            if tensor.shape[dimension] != size * tp_size:
                raise ValueError(f"Full TP source size does not match native topology for {target}")
            expected = tensor.narrow(dimension, tp_rank * size, size)
            actual, kind = parameter, f"tp_dimension_{dimension}"
        if expected.shape != actual.shape:
            raise ValueError(f"Full native local slice shape mismatch for {target}")
        return expected, actual, kind

    def load_bucket(self, weights):
        items = list(weights)
        self.bucket_count += 1
        self.maximum_bucket_bytes = max(self.maximum_bucket_bytes,
                                        sum(tensor.numel() * tensor.element_size() for _, tensor in items))
        expected = self.metadata["tensors"]
        targets = []
        for source, tensor in items:
            if source in self.source or source not in expected:
                raise ValueError(f"Duplicate or unexpected Full source tensor: {source}")
            shape_dtype = {"shape": list(tensor.shape), "dtype": str(tensor.dtype)}
            if shape_dtype != expected[source]:
                raise ValueError(f"Full source shape/dtype changed: {source}")
            self.source[source] = shape_dtype
            target, part = self._target(source)
            targets.append((source, target, part, tensor))
            if part is not None:
                self.parts.setdefault(target, set()).add(part)
            if source in self.metadata["sample_names"]:
                self.samples[source] = tensor_sample(tensor)
        loaded = self.model.load_weights([(source.removeprefix("transformer."), tensor) for source, tensor in items])
        if not isinstance(loaded, set) or not {target for _, target, _, _ in targets} <= loaded:
            raise RuntimeError("Native Full loader did not acknowledge every mapped target in bucket.")
        self.loaded.update(loaded)
        for source, target, part, tensor in targets:
            expected, actual, kind = self._local_slice(target, part, tensor)
            expected_sample = tensor_sample(expected.to(dtype=actual.dtype))
            actual_sample = tensor_sample(actual)
            if actual_sample != expected_sample:
                raise RuntimeError(f"Native Full loader acknowledged but did not copy {source}")
            self.post_load_checks += 1
            if source in self.metadata["sample_names"] or kind not in self.checked_kinds:
                self.post_load_samples[source] = {"target": target, "kind": kind, "actual": actual_sample,
                                                  "expected": expected_sample}
            self.checked_kinds.add(kind)

    def finish(self):
        if set(self.source) != set(self.metadata["tensors"]):
            raise RuntimeError("Full weight stream is incomplete.")
        if self.loaded != set(self.parameters):
            raise RuntimeError(f"Full native transformer coverage incomplete: {set(self.parameters) - self.loaded}")
        for target, parts in self.parts.items():
            expected_parts = {shard_id for parameter_name, _, shard_id in self.model.stacked_params_mapping
                              if parameter_name in target}
            if parts != expected_parts:
                raise RuntimeError(f"Incomplete QKV/TP source components for {target}")
        if not self.post_load_checks:
            raise RuntimeError("No post-load Full transformer weights were independently checked.")
        return {"policy_version": self.metadata["policy_version"], "global_steps": self.metadata["global_steps"],
                **{key: self.metadata[key] for key in (
                    "policy_algorithm", "optimizer_updates_per_rollout"
                ) if key in self.metadata},
                "source_tensors": self.source, "loaded_targets": sorted(self.loaded),
                "qkv_parts": {name: sorted(parts, key=str) for name, parts in self.parts.items()},
                "source_samples": self.samples, "post_load_checks": self.post_load_checks,
                "post_load_samples": self.post_load_samples, "checked_kinds": sorted(self.checked_kinds),
                "buckets": self.bucket_count, "maximum_bucket_bytes": self.maximum_bucket_bytes}
