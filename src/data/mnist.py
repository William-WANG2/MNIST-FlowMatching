from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def build_train_loader(
    data_root: Path,
    batch_size: int,
    image_size: int,
    mean: float,
    std: float,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader:
    dataset = datasets.MNIST(
        root=str(data_root),
        train=True,
        download=True,
        transform=transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize((mean,), (std,)),
            ]
        ),
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


class InfiniteBatchIterator:
    def __init__(self, dataloader: DataLoader):
        self.dataloader = dataloader
        self.iterator = iter(dataloader)

    def next(self):
        try:
            return next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.dataloader)
            return next(self.iterator)

