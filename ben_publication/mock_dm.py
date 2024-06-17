import torch
from lightning.pytorch import LightningDataModule
from torch.utils.data.dataset import Dataset


class MockDataset(Dataset):
    def __init__(self, dims, clss, length):
        self.data = torch.rand(*dims)
        self.targets = torch.rand(clss)
        self.length = length

    def __getitem__(self, index):
        return self.data, self.targets

    def __len__(self):
        return self.length


class MockDataModule(LightningDataModule):
    def __init__(self, dims, clss, train_length, val_length, test_length, bs, num_workers=8):
        super().__init__()
        self.dims = dims
        self.clss = clss
        self.length = train_length
        self.val_length = val_length
        self.test_length = test_length
        self.batch_size = bs
        self.num_workers = num_workers
        self.train_ds = MockDataset(self.dims, self.clss, self.length)
        self.val_ds = MockDataset(self.dims, self.clss, self.val_length)
        self.test_ds = MockDataset(self.dims, self.clss, self.test_length)
        print("MockDataModule initialized")

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_ds, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_ds, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_ds, batch_size=self.batch_size, num_workers=self.num_workers)
