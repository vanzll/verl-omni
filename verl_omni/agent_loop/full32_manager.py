import asyncio

import numpy as np
from verl.experimental.agent_loop.agent_loop import AgentLoopManager
from verl.protocol import DataProto
from verl.utils.ray_utils import auto_await


class Full32AgentLoopManager(AgentLoopManager):
    @auto_await
    async def generate_sequences(self, prompts: DataProto) -> DataProto:
        if not prompts.meta_info.get("validate", False):
            return await super().generate_sequences(prompts)
        if not self.model_config.full_finetune or len(prompts) != 8:
            raise ValueError("Full32 validation requires exactly the eight real validation cases.")
        if "priority" not in prompts.non_tensor_batch:
            prompts.non_tensor_batch["priority"] = np.arange(len(prompts), dtype=np.int64)
        workers = self.agent_loop_workers[:len(prompts)]
        if len(workers) != len(prompts):
            raise RuntimeError("Full32 validation needs one agent worker per real case, without padding.")
        chunks = prompts.chunk(len(workers))
        outputs = await asyncio.gather(*[worker.generate_sequences.remote(chunk)
                                        for worker, chunk in zip(workers, chunks, strict=True)])
        output = DataProto.concat(outputs)
        metrics = [item.meta_info.pop("metrics") for item in outputs]
        output.meta_info = {"timing": self._performance_metrics(metrics, output), **outputs[0].meta_info}
        return output
