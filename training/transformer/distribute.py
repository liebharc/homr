# S101: assert is intentional for distributed init preconditions
# flake8: noqa: S101

import os

import torch


class Distribute:
    def __init__(self) -> None:
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        if world_size <= 1:
            self.enabled = False
            return

        assert torch.cuda.is_available()
        assert torch.distributed.is_available()
        assert not torch.distributed.is_initialized()

        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)
        self._device = torch.device(f"cuda:{local_rank}")
        torch.distributed.init_process_group(backend="nccl", device_id=self._device)
        self.enabled = True
        self.rank = torch.distributed.get_rank()
        self.world_size = world_size

    def barrier(self) -> None:
        if not self.enabled:
            return
        torch.distributed.barrier(device_ids=[self._device.index])

    def destroy(self) -> None:
        if not self.enabled:
            return
        torch.distributed.destroy_process_group()

    def is_rank0(self) -> bool:
        if not self.enabled:
            return True
        return self.rank == 0

    def get_world_size(self) -> int:
        if not self.enabled:
            return 1
        return self.world_size
