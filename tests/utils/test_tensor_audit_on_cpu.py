import importlib.util
from pathlib import Path
import unittest

import torch


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("full_tensor_audit_cpu_test", ROOT / "verl_omni/utils/tensor_audit.py")
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


class TensorAuditTests(unittest.TestCase):
    def test_exact_integer_indices_at_full_model_scale(self):
        for count in (0, 1, 2, 7, 8, 9, 2**24, 2**24 + 1, 2**25, 2**26, 2**31, 2**60):
            with self.subTest(numel=count):
                indices = AUDIT.tensor_sample_indices(count)
                self.assertEqual(len(indices), min(8, count))
                self.assertEqual(indices, sorted(indices))
                self.assertTrue(all(0 <= index < count for index in indices))
                if count:
                    self.assertEqual(indices[0], 0)
                    self.assertEqual(indices[-1], count - 1)
        with self.assertRaises(ValueError):
            AUDIT.tensor_sample_indices(-1)

    def test_old_fp32_linspace_reproduces_out_of_bounds_without_large_allocation(self):
        count = 2**26
        tensor = torch.tensor([2.5], dtype=torch.bfloat16).expand(count)
        self.assertEqual(tensor.untyped_storage().nbytes(), 2)
        old_indices = torch.linspace(0, count - 1, 8).long()
        self.assertEqual(int(old_indices[-1]), count)
        with self.assertRaises(IndexError):
            tensor[old_indices]
        sample = AUDIT.tensor_sample(tensor)
        self.assertEqual(sample["flat_indices"][-1], count - 1)
        self.assertEqual(sample["values"], [2.5] * 8)

    def test_samples_empty_singleton_noncontiguous_and_preserves_dtype(self):
        for tensor in (torch.empty(0), torch.tensor([3.0], dtype=torch.bfloat16), torch.arange(24).reshape(4, 6).T):
            with self.subTest(shape=tuple(tensor.shape)):
                sample = AUDIT.tensor_sample(tensor)
                self.assertEqual(sample["shape"], list(tensor.shape))
                self.assertEqual(sample["dtype"], str(tensor.dtype))
                self.assertEqual(sample["values"], [float(tensor.reshape(-1)[index]) for index in sample["flat_indices"]])


if __name__ == "__main__":
    unittest.main()
