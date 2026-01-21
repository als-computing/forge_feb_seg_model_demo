import torch
from torch.utils.data import DataLoader, TensorDataset, DistributedSampler


def create_inference_dataloader(patched_images, batch_size, num_workers=2, pin_memory=True) -> DataLoader:
    """Create DataLoader for patched images."""

    dataset = TensorDataset(patched_images)

    inference_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=pin_memory,
        num_workers=num_workers,
    )

    return inference_loader


def create_distributed_dataloader(
    patches: torch.Tensor,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    world_size: int,
    rank: int
) -> DataLoader:
    """
    Create a distributed dataloader with DistributedSampler.
    
    Args:
        patches: Tensor of patches to process
        batch_size: Batch size per GPU
        num_workers: Number of workers for dataloader
        pin_memory: Whether to pin memory
        world_size: Total number of processes
        rank: Current process rank
        
    Returns:
        DataLoader with distributed sampling
    """
    # Create a simple dataset wrapper
    class TensorDataset(torch.utils.data.Dataset):
        def __init__(self, tensor):
            self.tensor = tensor
        
        def __len__(self):
            return len(self.tensor)
        
        def __getitem__(self, idx):
            return self.tensor[idx]
    
    dataset = TensorDataset(patches)
    
    if world_size > 1:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
            drop_last=False
        )
    else:
        sampler = None
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False if sampler else False,
    )
    
    return dataloader