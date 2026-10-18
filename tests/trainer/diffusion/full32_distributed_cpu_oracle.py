import ast
from datetime import timedelta
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Shard, distribute_tensor


def native_clip_function():
    import importlib.util

    spec = importlib.util.find_spec("verl")
    path = Path(next(iter(spec.submodule_search_locations))) / "utils/fsdp_utils.py"
    tree = ast.parse(path.read_text())
    method = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                  and node.name == "fsdp2_clip_grad_norm_")
    namespace = {"torch": torch, "get_device_id": lambda: "cpu"}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[method.name]


def global_value(value):
    return value.full_tensor() if isinstance(value, DTensor) else value


def run():
    torch.set_num_threads(1)
    dist.init_process_group("gloo", timeout=timedelta(seconds=100))
    rank = dist.get_rank()
    world = dist.get_world_size()
    if world != 4:
        raise ValueError("CPU ownership/clip oracle requires four ranks.")
    mesh = init_device_mesh("cpu", (world,), mesh_dim_names=("fsdp",))
    clip = native_clip_function()
    initial = torch.linspace(-0.2, 0.3, 32).reshape(8, 4)
    inputs = torch.linspace(-1.0, 2.0, 64).reshape(16, 4)
    targets = torch.linspace(-0.4, 0.5, 128).reshape(16, 8)
    results = []
    for microbatch in (4, 2, 1):
        parameter = torch.nn.Parameter(distribute_tensor(initial.clone(), mesh, [Shard(0)]))
        dense = torch.nn.Parameter(initial.clone())
        kwargs = {"lr": 3e-5, "betas": (0.9, 0.999), "weight_decay": 1e-4, "foreach": False}
        optimizer = torch.optim.AdamW([parameter], **kwargs)
        reference = torch.optim.AdamW([dense], **kwargs)
        norms = []
        for update in range(2):
            optimizer.zero_grad()
            reference.zero_grad()
            full = parameter.full_tensor().detach().requires_grad_(True)
            local_inputs = inputs[rank * 4:(rank + 1) * 4] + update * 0.03
            local_targets = targets[rank * 4:(rank + 1) * 4]
            for start in range(0, 4, microbatch):
                for timestep in range(19):
                    prediction = (local_inputs[start:start + microbatch] + timestep * 0.001) @ full.T
                    loss = (prediction - local_targets[start:start + microbatch]).square().mean()
                    (loss / (19 * 4 / microbatch)).backward()
            dist.all_reduce(full.grad)
            full.grad.div_(world)
            local_gradient = full.grad.chunk(world, dim=0)[rank].contiguous()
            parameter.grad = DTensor.from_local(local_gradient, mesh, [Shard(0)], run_check=True)
            for timestep in range(19):
                prediction = (inputs + update * 0.03 + timestep * 0.001) @ dense.T
                ((prediction - targets).square().mean() / 19).backward()
            torch.testing.assert_close(parameter.grad.full_tensor(), dense.grad, rtol=2e-5, atol=2e-7)
            actual_norm = clip([parameter], 0.01, foreach=False)
            expected_norm = torch.nn.utils.clip_grad_norm_([dense], 0.01, foreach=False)
            torch.testing.assert_close(global_value(actual_norm), expected_norm, rtol=2e-5, atol=2e-7)
            if float(global_value(actual_norm)) <= 0.01:
                raise AssertionError("Oracle must exercise active global clipping.")
            torch.testing.assert_close(parameter.grad.full_tensor(), dense.grad, rtol=2e-5, atol=2e-8)
            optimizer.step()
            reference.step()
            torch.testing.assert_close(parameter.full_tensor(), dense, rtol=2e-6, atol=2e-8)
            for key in ("exp_avg", "exp_avg_sq"):
                state = optimizer.state[parameter][key]
                if not isinstance(state, DTensor) or state.to_local().numel() != state.numel() // world:
                    raise AssertionError("Adam moments must be genuinely sharded.")
                torch.testing.assert_close(state.full_tensor(), reference.state[dense][key], rtol=3e-5, atol=1e-10)
            norms.append(float(global_value(actual_norm)))
        results.append({"microbatch": microbatch, "accumulation": 4 // microbatch, "timesteps": 19,
                        "updates": 2, "global_norms": norms, "local_shape": list(parameter.to_local().shape),
                        "global_shape": list(parameter.shape), "placements": [str(item) for item in parameter.placements]})
    print(json.dumps({"rank": rank, "world": world, "device": "cpu", "oracle": results}), flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    run()
