import ast
from contextlib import contextmanager, nullcontext
import importlib.util
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import warnings

import torch


ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = ROOT / "verl_omni/workers/engine/full_policy.py"
SPEC = importlib.util.spec_from_file_location("full_policy_cpu_test", MODULE_PATH)
POLICY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(POLICY)


class LegacyPolicy:
    def copy_adapter(self, source="default", target="old"):
        return ("legacy_copy", source, target)

    def ema_update_adapter(self, source="default", target="old", decay=0.0):
        return ("legacy_ema", decay)

    @contextmanager
    def use_adapter(self, name):
        self.legacy_name = name
        yield


class TinyFullEngine(POLICY.FullPolicyMixin, LegacyPolicy):
    def __init__(self, full=True):
        self.model_config = SimpleNamespace(full_finetune=full, lora_rank=0 if full else 32,
                                            lora_adapter_path=None, algorithm="diffusion_art")
        self.optimizer_config = SimpleNamespace(clip_grad=0.01)
        self.module = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.SiLU(), torch.nn.Linear(4, 2))
        self.optimizer = torch.optim.AdamW(self.module.parameters(), lr=3e-5, betas=(0.9, 0.999),
                                          weight_decay=1e-4, foreach=False)
        self.events = []
        self._initialize_full_policy()

    def _full_event(self, kind, **values):
        self.events.append((kind, self._full_policy_version, self._full_old_version, values))

    def update(self, inputs, target, microbatch):
        self.optimizer.zero_grad()
        for start in range(0, len(inputs), microbatch):
            for timestep in range(19):
                prediction = self.module(inputs[start:start + microbatch] + timestep * 0.001)
                loss = (prediction - target[start:start + microbatch]).square().mean()
                (loss / (19 * len(inputs) / microbatch)).backward()
        norm = torch.nn.utils.clip_grad_norm_(self.module.parameters(), self.optimizer_config.clip_grad)
        before = self._full_parameter_samples()
        self.optimizer.step()
        self._full_step_applied(before, norm)


class FullPolicyTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def test_full_has_no_peft_and_covers_every_parameter(self):
        engine = TinyFullEngine()
        self.assertFalse(hasattr(engine.module, "peft_config"))
        audit = POLICY.parameter_ownership(engine.module, engine.optimizer)
        self.assertEqual(audit["global_parameter_numel"], sum(parameter.numel() for parameter in engine.module.parameters()))
        self.assertEqual(audit["parameter_tensors"], audit["optimizer_parameter_tensors"])
        engine.module[0].weight.requires_grad_(False)
        with self.assertRaises(ValueError):
            POLICY.parameter_ownership(engine.module, engine.optimizer)

    def test_full_rejects_missing_optimizer_parameters(self):
        engine = TinyFullEngine()
        engine.optimizer.param_groups[0]["params"].pop()
        with self.assertRaises(ValueError):
            POLICY.parameter_ownership(engine.module, engine.optimizer)

    def test_legacy_adapter_behavior_is_not_a_noop(self):
        engine = TinyFullEngine(full=False)
        self.assertEqual(engine.copy_adapter(), ("legacy_copy", "default", "old"))
        self.assertEqual(engine.ema_update_adapter(decay=0.7), ("legacy_ema", 0.7))
        with engine.use_adapter("old"):
            self.assertEqual(engine.legacy_name, "old")

    def test_two_updates_keep_old_frozen_then_commit_boundary(self):
        engine = TinyFullEngine()
        inputs = torch.randn(4, 3)
        cached_target = torch.randn(4, 2)
        original_target = cached_target.clone()
        engine.copy_adapter()
        self.assertEqual(engine._full_export_version("old"), 0)
        engine.update(inputs, cached_target, 4)
        self.assertEqual(engine._full_old_version, 0)
        with self.assertRaises(ValueError):
            engine.copy_adapter()
        with self.assertRaises(ValueError):
            engine._full_export_version("old")
        engine.update(inputs, cached_target, 4)
        self.assertEqual(engine._full_old_version, 0)
        torch.testing.assert_close(original_target, cached_target, rtol=0, atol=0)
        engine.copy_adapter()
        self.assertEqual(engine._full_export_version("old"), 2)
        self.assertGreater(engine._full_last_step["global_changed_parameter_samples"], 0)

    def test_unknown_policy_and_old_forward_fail_closed(self):
        engine = TinyFullEngine()
        engine.copy_adapter()
        for name in (None, "default", "reference", "typo"):
            with self.assertRaises(ValueError):
                engine._full_export_version(name)
        for name in ("old", "reference", "typo"):
            with self.assertRaises(ValueError), engine.use_adapter(name):
                pass
        with self.assertRaises(ValueError):
            engine.ema_update_adapter(decay=0.9)

    def test_microbatch_accumulation_and_clipped_adam_equivalence(self):
        reference = TinyFullEngine()
        initial = {name: parameter.clone() for name, parameter in reference.module.state_dict().items()}
        inputs = torch.randn(4, 3)
        target = torch.randn(4, 2)
        candidates = []
        for microbatch in (4, 2, 1):
            engine = TinyFullEngine()
            engine.module.load_state_dict(initial)
            for update in range(2):
                engine.update(inputs + update * 0.03, target, microbatch)
            self.assertLess(engine._full_last_step["clip_coefficient"], 1)
            candidates.append(engine)
        for engine in candidates[1:]:
            for reference_parameter, actual_parameter in zip(candidates[0].module.parameters(), engine.module.parameters(), strict=True):
                torch.testing.assert_close(reference_parameter, actual_parameter, rtol=1e-6, atol=1e-7)
                for name in ("exp_avg", "exp_avg_sq"):
                    torch.testing.assert_close(candidates[0].optimizer.state[reference_parameter][name],
                                               engine.optimizer.state[actual_parameter][name], rtol=1e-5, atol=1e-9)

    def test_full_export_is_lazy_and_keeps_dtype(self):
        path = ROOT / "verl_omni/workers/engine/fsdp/diffusers_impl.py"
        tree = ast.parse(path.read_text())
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "DiffusersFSDPEngine")
        method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "get_per_tensor_param")
        events = []

        class FakeShard:
            def __init__(self, tensor):
                self.tensor = tensor
                self.shape = tensor.shape
                self.dtype = tensor.dtype

            def to(self, device):
                return self

            def full_tensor(self):
                events.append("gather")
                return self.tensor

        namespace = {"DTensor": FakeShard, "convert_weight_keys": lambda values, module: values,
                     "get_device_id": lambda: "cpu", "tensor_sample": lambda tensor: tensor.tolist()}
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), namespace)
        engine = TinyFullEngine()
        engine.copy_adapter()
        tensors = {"first": FakeShard(torch.ones(2, dtype=torch.float32)),
                   "second": FakeShard(torch.ones(3, dtype=torch.bfloat16))}
        engine.module = SimpleNamespace(state_dict=lambda: tensors)
        iterator, metadata = namespace["get_per_tensor_param"](engine, adapter_name="old")
        self.assertIsNone(metadata)
        self.assertFalse(events)
        name, tensor = next(iterator)
        self.assertEqual((name, tensor.dtype), ("transformer.first", torch.float32))
        self.assertEqual(len(events), 1)
        engine._full_policy_version = 1
        with self.assertRaises(RuntimeError):
            next(iterator)

    def test_full_cache_version_is_frozen_across_windows(self):
        engine = TinyFullEngine()
        engine.copy_adapter()
        data = {"art_policy_version": torch.zeros(4, dtype=torch.int64),
                "art_update_window": torch.zeros(4, dtype=torch.int64),
                "train_timesteps": torch.ones(4, 19)}
        engine._validate_full_cached_window(data, "train_timesteps")
        engine._full_policy_version = 1
        data["art_update_window"].fill_(1)
        engine._validate_full_cached_window(data, "train_timesteps")
        data["art_policy_version"].fill_(1)
        with self.assertRaises(ValueError):
            engine._validate_full_cached_window(data, "train_timesteps")

        data["art_policy_version"].zero_()
        data["train_timesteps"] = torch.ones(4, 18)
        with self.assertRaises(ValueError):
            engine._validate_full_cached_window(data, "train_timesteps")

    def test_only_full_rank_zero_loads_pretrained_others_stay_meta(self):
        path = ROOT / "verl_omni/workers/engine/fsdp/diffusers_impl.py"
        tree = ast.parse(path.read_text())
        method = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "_build_module")
        loaded = []
        created = []

        class Model(torch.nn.Linear):
            def __init__(self):
                super().__init__(3, 2)
                self.config = SimpleNamespace()

            def set_attention_backend(self, backend):
                self.backend = backend

        class LTXClass:
            @staticmethod
            def load_config(*args, **kwargs):
                return {}

            @staticmethod
            def from_config(config):
                model = Model()
                created.append(model)
                return model

        def load(*args, **kwargs):
            model = Model()
            loaded.append(model)
            return model

        namespace = {"torch": torch, "os": os, "warnings": warnings,
                     "get_init_weight_context_manager": lambda **kwargs: nullcontext,
                     "_load_diffusers_pretrained": load,
                     "_cast_loaded_diffusers_module": lambda module, dtype, **kwargs: module.to(dtype),
                     "DiffusionModelBase": SimpleNamespace(get_class=lambda config:
                         SimpleNamespace(preserve_fp32_modules=lambda: ())) }
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), namespace)
        with patch.dict(sys.modules, {"diffusers": SimpleNamespace(LTX2VideoTransformer3DModel=LTXClass)}):
            for full, rank in ((True, 0), (True, 1), (False, 1)):
                engine = SimpleNamespace(full_finetune=full, rank=rank, device_mesh=None,
                    model_config=SimpleNamespace(config_path=None, local_path="same-pretrained", transformer_subfolder="transformer",
                                                  trust_remote_code=False, attn_backend="unchanged", enable_gradient_checkpointing=False),
                    engine_config=SimpleNamespace(model_dtype="bfloat16", forward_only=False),
                    _build_module_from_registry=lambda dtype: None, _full_event=lambda *args, **kwargs: None)
                model = namespace[method.name](engine)
                self.assertEqual(next(model.parameters()).is_meta, full and rank != 0)
                self.assertEqual(next(model.parameters()).dtype, torch.bfloat16)
        self.assertEqual(len(loaded), 2)
        self.assertEqual(len(created), 1)


if __name__ == "__main__":
    unittest.main()
