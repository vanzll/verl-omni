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

import math

import torch

from verl_omni.workers.engine.full_policy import FullPolicyMixin, local_tensor


class FullNFTPolicyMixin(FullPolicyMixin):
    @property
    def full_nft(self):
        return self.full_finetune and self.model_config.algorithm == "omni_nft"

    def _initialize_full_policy(self):
        if not self.full_nft:
            return super()._initialize_full_policy()
        if self.model_config.lora_rank != 0 or self.model_config.lora_adapter_path is not None:
            raise ValueError("Full NFT cannot construct or initialize LoRA adapters.")
        self._full_policy_version = 0
        self._full_old_version = None
        self._full_recent_updates = []
        self._nft_old_module = None
        self._nft_ema_updates = 0
        self._nft_ema_rollouts = 0
        from verl_omni.utils.full_policy_sync import full_policy_sync_tag

        self._nft_updates_per_rollout = full_policy_sync_tag("omni_nft")["optimizer_updates_per_rollout"]

    def _nft_parameter_pairs(self):
        if self._nft_old_module is None:
            raise RuntimeError("Full NFT requires a distinct frozen EMA policy module.")
        current = dict(self.module.named_parameters())
        old = dict(self._nft_old_module.named_parameters())
        if current.keys() != old.keys():
            raise RuntimeError("Full NFT current/EMA parameter identities differ.")
        pairs = []
        for name, parameter in current.items():
            target = old[name]
            source_local, target_local = local_tensor(parameter), local_tensor(target)
            if parameter.shape != target.shape or source_local.shape != target_local.shape or parameter.dtype != target.dtype:
                raise RuntimeError(f"Full NFT current/EMA shard shapes differ: {name}")
            if (source_local.numel() and source_local.data_ptr() == target_local.data_ptr()) or target.requires_grad or target.grad is not None:
                raise RuntimeError(f"Full NFT EMA must be frozen and non-aliasing: {name}")
            if hasattr(parameter, "placements") and (
                parameter.placements != target.placements
                or not torch.equal(parameter.device_mesh.mesh, target.device_mesh.mesh)
            ):
                raise RuntimeError(f"Full NFT current/EMA shard ownership differs: {name}")
            pairs.append((name, parameter, target))
        return pairs

    def _nft_policy_summary(self):
        pairs = self._nft_parameter_pairs()
        return {
            "policy_kind": "nft_current_and_frozen_ema",
            "ema_parameter_tensors": len(pairs),
            "ema_global_parameter_numel": sum(target.numel() for _, _, target in pairs),
            "ema_local_parameter_numel": sum(local_tensor(target).numel() for _, _, target in pairs),
            "ema_frozen_nonaliasing": True,
            "base_reference_constructed": False,
            "ema_updates": self._nft_ema_rollouts,
            "ema_policy_version": self._nft_ema_updates,
            "optimizer_updates_per_rollout": self._nft_updates_per_rollout,
        }

    def copy_adapter(self, source="default", target="old"):
        if not self.full_nft:
            return super().copy_adapter(source=source, target=target)
        return self.ema_update_adapter(source=source, target=target, decay=0.0)

    def ema_update_adapter(self, source="default", target="old", decay=0.0):
        if not self.full_nft:
            return super().ema_update_adapter(source=source, target=target, decay=decay)
        if (source, target) != ("default", "old") or not math.isfinite(decay) or not 0.0 <= decay <= 1.0:
            raise ValueError("Full NFT supports only finite default-to-old EMA decay in [0,1].")
        if self._full_policy_version != self._nft_ema_updates:
            if self._full_policy_version != self._nft_ema_updates + self._nft_updates_per_rollout:
                raise RuntimeError("NFT EMA must run once after all successful updates in the rollout.")
        elif self._full_policy_version != 0 or self._full_old_version is not None:
            raise RuntimeError("NFT EMA was repeated without a new optimizer update.")
        before = []
        expected = []
        actual = []
        with torch.no_grad():
            for _, parameter, target in self._nft_parameter_pairs():
                source_local, target_local = local_tensor(parameter), local_tensor(target)
                if source_local.numel():
                    sample = source_local.numel() // 2
                    previous = target_local.reshape(-1)[sample].clone()
                    current = source_local.reshape(-1)[sample]
                    before.append(previous.float())
                    expected.append(torch.lerp(previous, current, 1.0 - decay).float())
                target_local.lerp_(source_local, 1.0 - decay)
                if source_local.numel():
                    actual.append(target_local.reshape(-1)[sample].float())
        expected_samples = torch.stack(expected)
        actual_samples = torch.stack(actual)
        if not torch.equal(expected_samples, actual_samples):
            raise RuntimeError("NFT sharded EMA differs from the original lerp update.")
        if self._full_policy_version:
            self._nft_ema_rollouts += 1
        self._full_old_version = self._full_policy_version
        self._nft_ema_updates = self._full_policy_version
        self._full_event(
            "nft_ema_update", decay=float(decay), successful_optimizer_updates=self._full_policy_version,
            changed_local_samples=int((torch.stack(before) != actual_samples).sum().item()),
            expected_sample_max_error=float((expected_samples - actual_samples).abs().max().item()),
            **self._nft_policy_summary(),
        )

    def _full_export_version(self, adapter_name):
        if not self.full_nft:
            return super()._full_export_version(adapter_name)
        if adapter_name != "old" or self._full_old_version != self._full_policy_version:
            raise ValueError("Full NFT rollout export requires the committed EMA-old policy.")
        self._nft_parameter_pairs()
        return self._full_old_version

    def _validate_full_cached_window(self, data, timesteps_key):
        if not self.full_nft:
            return super()._validate_full_cached_window(data, timesteps_key)
        window = self._full_policy_version - self._nft_ema_updates
        local_samples = 8 // self._nft_updates_per_rollout
        if (self._full_old_version != self._nft_ema_updates
                or not 0 <= window < self._nft_updates_per_rollout or len(data) != local_samples):
            raise ValueError("Full NFT window must match the frozen EMA rollout boundary and local minibatch.")
        if data[timesteps_key].shape[1] != 20:
            raise ValueError("Full NFT requires all twenty original training timesteps.")
        if any(key in data for key in ("art_update_window", "train_conditional_velocities")):
            raise ValueError("NFT must not consume ART windows or rollout velocity caches.")
        self._nft_parameter_pairs()
        self._full_event("nft_window", samples=local_samples, training_timesteps=20,
                         update_window=window,
                         target_source="fresh_forward_noised_state_and_frozen_ema_old")

    def full_policy_state(self):
        state = super().full_policy_state()
        if self.full_nft:
            state.update(self._nft_policy_summary())
        return state
