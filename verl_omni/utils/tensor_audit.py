import torch


def tensor_sample_indices(numel):
    if numel < 0:
        raise ValueError("Tensor element count cannot be negative.")
    count = min(8, numel)
    return [index * (numel - 1) // max(1, count - 1) for index in range(count)]


def tensor_sample(tensor):
    flat = tensor.detach().reshape(-1)
    flat_indices = tensor_sample_indices(flat.numel())
    indices = torch.tensor(flat_indices, dtype=torch.int64, device=flat.device)
    return {"shape": list(tensor.shape), "dtype": str(tensor.dtype),
            "flat_indices": flat_indices, "values": flat[indices].float().cpu().tolist()}
