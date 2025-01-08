import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms


class NewDataset(Dataset):
    def __init__(self, file, ds_name, window_length):
        super(NewDataset, self).__init__()
        self.transform = torch.nn.Sequential(
            transforms.Resize((36, 36))
        )

        if ds_name == 'MANHOB_HCI':
            self.add = 2
        else:
            self.add = 1

        self.video = np.asarray(file["preprocessed_video"])
        self.label = np.asarray(file["preprocessed_label"])
        self.tot_length = len(self.label)

        self.window_length = window_length

    def __len__(self):
        return (self.tot_length - self.add) // self.window_length

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        seq_i = idx * self.window_length

        x = torch.tensor(self.video[seq_i:seq_i + self.window_length + self.add], dtype=torch.float32).permute(0, 3, 1, 2)
        y = torch.tensor(self.label[seq_i:seq_i + self.window_length], dtype=torch.float32)

        x = self.transform(x)
        return x, y
