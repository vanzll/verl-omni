import ast
import copy
import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location("full32_gate_cpu_test", ROOT / "verl_omni/utils/full32_gate.py")
GATE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GATE)


def records():
    result = []
    for rank in range(32):
        actor = {"rank": rank, "hostname": f"host{rank // 8}", "strategy": "fsdp2", "fsdp_size": 32,
                 "sample_dp_size": 32, "mesh_ranks": [list(range(32))], "policy_version": 2, "old_version": 2,
                 "export_version": 2, "global_parameter_numel": 64, "parameter_tensors": 4,
                 "optimizer_parameter_tensors": 4, "local_parameter_numel": 2, "local_optimizer_state_bytes": 16,
                 "recent_updates": [{"policy_version": 1, "global_preclip_norm": 0.4},
                                    {"policy_version": 2, "global_preclip_norm": 0.5}]}
        rollout = None
        if rank % 2 == 0:
            rollout = [[{"ready": True, "hostname": f"host{(rank + offset) // 8}", "physical_gpu": str((rank + offset) % 8),
                         "state": {"policy_version": 2, "global_steps": 1, "initial_sync_completed": True,
                                   "source_tensor_count": 4, "post_load_checks": 4,
                                   "checked_kinds": ["packed_q", "packed_k", "packed_v"],
                                   "fixed_input_output_change": {"video": {"changed_elements": 1}}}}
                        for offset in range(2)]]
        result.append({"actor": actor, "rollout": rollout})
    return result


class Full32GateTests(unittest.TestCase):
    def test_nft_requires_one_actual_update_and_distinct_frozen_ema(self):
        actual = records()
        for record in actual:
            actor = record["actor"]
            actor.update(policy_version=1, old_version=1, export_version=1,
                         policy_kind="nft_current_and_frozen_ema", ema_frozen_nonaliasing=True,
                         ema_updates=1, ema_policy_version=1, optimizer_updates_per_rollout=1,
                         ema_parameter_tensors=4, ema_global_parameter_numel=64,
                         ema_local_parameter_numel=2, base_reference_constructed=False)
            actor["recent_updates"] = actor["recent_updates"][:1]
            if record["rollout"]:
                for receiver in record["rollout"][0]:
                    receiver["state"].update(policy_version=1, policy_algorithm="omni_nft",
                                             optimizer_updates_per_rollout=1)
        evidence = GATE.validate_full32_gate(actual, expected_parameters=64, expected_tensors=4)
        self.assertIn("nft_u1", evidence["status"])
        for key, value in (("ema_frozen_nonaliasing", False), ("ema_updates", 0),
                           ("base_reference_constructed", True), ("ema_local_parameter_numel", 64)):
            broken = copy.deepcopy(actual)
            broken[0]["actor"][key] = value
            with self.assertRaises(RuntimeError):
                GATE.validate_full32_gate(broken, expected_parameters=64, expected_tensors=4)

    def test_nft_u2_requires_two_real_steps_but_one_ema_refresh(self):
        actual = records()
        for record in actual:
            record["actor"].update(
                policy_kind="nft_current_and_frozen_ema", ema_frozen_nonaliasing=True,
                ema_updates=1, ema_policy_version=2, optimizer_updates_per_rollout=2,
                ema_parameter_tensors=4, ema_global_parameter_numel=64,
                ema_local_parameter_numel=2, base_reference_constructed=False,
            )
            if record["rollout"]:
                for receiver in record["rollout"][0]:
                    receiver["state"].update(policy_algorithm="omni_nft", optimizer_updates_per_rollout=2)
        evidence = GATE.validate_full32_gate(actual, expected_parameters=64, expected_tensors=4)
        self.assertIn("nft_u2", evidence["status"])
        for key, value in (("ema_updates", 2), ("ema_policy_version", 1),
                           ("optimizer_updates_per_rollout", 1), ("policy_kind", "art"),
                           ("recent_updates", [{"policy_version": 2, "global_preclip_norm": 0.4}])):
            with self.subTest(actor_field=key):
                broken = copy.deepcopy(actual)
                broken[0]["actor"][key] = value
                with self.assertRaises(RuntimeError):
                    GATE.validate_full32_gate(broken, expected_parameters=64, expected_tensors=4)
        for key, value in (("policy_algorithm", "diffusion_art"), ("optimizer_updates_per_rollout", 1),
                           ("policy_version", 1)):
            with self.subTest(native_field=key):
                broken = copy.deepcopy(actual)
                broken[0]["rollout"][0][0]["state"][key] = value
                with self.assertRaises(RuntimeError):
                    GATE.validate_full32_gate(broken, expected_parameters=64, expected_tensors=4)

    def test_complete_native_two_update_gate(self):
        evidence = GATE.validate_full32_gate(records(), expected_parameters=64, expected_tensors=4)
        self.assertEqual(len(evidence["actors"]), 32)
        self.assertEqual(len(evidence["native_receivers"]), 32)
        self.assertEqual(evidence["output_change_observed_receivers"], 32)

    def test_incomplete_or_wrong_rank_sharding_state_fails(self):
        for key, value in (("rank", 1), ("strategy", "fsdp"), ("fsdp_size", 1), ("sample_dp_size", 1),
                           ("mesh_ranks", [[0]]), ("old_version", 1), ("export_version", 0),
                           ("global_parameter_numel", 2), ("optimizer_parameter_tensors", 1),
                           ("local_parameter_numel", 64), ("local_optimizer_state_bytes", 0),
                           ("recent_updates", [{"policy_version": 1, "global_preclip_norm": 0.4}])):
            with self.subTest(field=key):
                actual = records()
                actual[0]["actor"][key] = value
                with self.assertRaises(RuntimeError):
                    GATE.validate_full32_gate(actual, expected_parameters=64, expected_tensors=4)

    def test_missing_or_duplicate_physical_receiver_and_partial_sync_fail(self):
        original = records()
        for key, value in (("policy_version", 0), ("global_steps", 0), ("initial_sync_completed", False),
                           ("source_tensor_count", 1), ("post_load_checks", 1), ("checked_kinds", ["replicated"])):
            with self.subTest(field=key):
                actual = copy.deepcopy(original)
                actual[0]["rollout"][0][0]["state"][key] = value
                with self.assertRaises(RuntimeError):
                    GATE.validate_full32_gate(actual, expected_parameters=64, expected_tensors=4)
        for missing in (True, False):
            actual = records()
            if missing:
                actual[0]["rollout"] = None
            else:
                actual[0]["rollout"][0][1]["physical_gpu"] = "0"
            with self.assertRaises(RuntimeError):
                GATE.validate_full32_gate(actual, expected_parameters=64, expected_tensors=4)

    def test_probe_nonchange_is_not_a_training_failure(self):
        actual = records()
        for record in actual:
            if record["rollout"]:
                for rank in record["rollout"][0]:
                    rank["state"]["fixed_input_output_change"] = None
        evidence = GATE.validate_full32_gate(actual, expected_parameters=64, expected_tensors=4)
        self.assertEqual(evidence["output_change_observed_receivers"], 0)

    def test_gate_is_on_actual_art_trainer_after_weight_sync_not_policy_gradient(self):
        path = ROOT / "verl_omni/trainer/diffusion/ray_diffusion_trainer.py"
        tree = ast.parse(path.read_text())
        for cls in (node for node in tree.body if isinstance(node, ast.ClassDef)):
            calls = [node for node in ast.walk(cls) if isinstance(node, ast.Call)
                     and isinstance(node.func, ast.Name) and node.func.id == "record_full32_startup_gate"]
            if cls.name == "DirectPreferenceRayTrainer":
                self.assertEqual(len(calls), 1)
                fit = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "fit")
                updates = [node for node in ast.walk(fit) if isinstance(node, ast.Call)
                           and isinstance(node.func, ast.Attribute) and node.func.attr == "update_weights"]
                self.assertLess(max(node.lineno for node in updates if node.lineno < calls[0].lineno), calls[0].lineno)
            else:
                self.assertFalse(calls)


if __name__ == "__main__":
    unittest.main()
