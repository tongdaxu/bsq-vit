import torchvision
from torch.utils.data import DataLoader
from torchvision.datasets import VisionDataset
from PIL import Image
from glob import glob


class InfiniteDataLoader:
    def __init__(self, data_loader):
        self.data_loader = data_loader
        self.data_iter = iter(data_loader)
        if not hasattr(self.data_loader.sampler, "epoch"):
            self.data_loader.sampler.epoch = 0

    def __iter__(self):
        return self

    def __next__(self):
        try:
            data = next(self.data_iter)
        except StopIteration:
            self.data_loader.sampler.epoch += 1
            self.data_iter = iter(self.data_loader)  # Reset the data loader
            data = next(self.data_iter)
        return data


class SimpleDataset(VisionDataset):
    def __init__(self, root: str, transform=None):
        super().__init__(root, transform)
        self.transform = transform
        if root.endswith(".txt"):
            with open(root) as f:
                lines = f.readlines()
            self.fpaths = [line.strip("\n") for line in lines]
        else:
            self.fpaths = sorted(glob(root + "/**/*.JPEG", recursive=True))
            self.fpaths += sorted(glob(root + "/**/*.jpg", recursive=True))
            self.fpaths += sorted(glob(root + "/**/*.png", recursive=True))

        assert len(self.fpaths) > 0, "File list is empty. Check the root."

    def __len__(self):
        return len(self.fpaths)

    def __getitem__(self, index: int):
        fpath = self.fpaths[index]
        img = Image.open(fpath).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, fpath
