import torch
from torch.utils.data import Dataset


class STMDataset_train(Dataset):
    def __init__(self, file, length, window_length):
        super(STMDataset_train, self).__init__()
        self.file = file
        self.len = length
        self.window_length = window_length

    def __len__(self):
        return (self.len - 1) // self.window_length

    def __getitem__(self, idx):
        # each time obtaining n=window_length frames from the file for 1 batch.
        if torch.is_tensor(idx):
            idx = idx.tolist()

        x = self.file['preprocessed_video'][idx * self.window_length:(idx * self.window_length) + self.window_length + 1]
        y = torch.squeeze(torch.tensor(self.file['preprocessed_label'][idx * self.window_length:(idx * self.window_length) + self.window_length], dtype=torch.float32))
        return x, y


class STMDataset_valid(Dataset):
    def __init__(self, file, length, train_length, window_length):
        super(STMDataset_valid, self).__init__()
        self.file = file
        self.len = length
        self.start_index = train_length
        self.window_length = window_length

    def __len__(self):
        return (self.len - 1) // self.window_length

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        x = self.file['preprocessed_video'][self.start_index + (idx * self.window_length):
                                            self.start_index + (idx * self.window_length) + self.window_length + 1]
        y = torch.squeeze(torch.tensor(self.file['preprocessed_label'][self.start_index + (idx * self.window_length):
                                                         self.start_index + (idx * self.window_length)
                                                         + self.window_length], dtype=torch.float32))

        return x, y


class STMDataset(Dataset):
    def __init__(self, file, ds_name, window_length):
        super(STMDataset, self).__init__()
        if ds_name in ["MANHOB_HCI", "VIPL"]:
            self.to_ram = False
        else:
            self.to_ram = True

        if self.to_ram:
            self.video = file["preprocessed_video"]
            self.label = file["preprocessed_label"]
            self.tot_length = len(self.label)
        else:
            self.file = file
            self.tot_length = len(self.file['preprocessed_label'])

        self.window_length = window_length

    def __len__(self):
        return (self.tot_length - 1) // self.window_length

    def __getitem__(self, idx):
        # each time obtaining n=window_length frames from the file for 1 batch.
        if torch.is_tensor(idx):
            idx = idx.tolist()

        seq_i = idx * self.window_length
        if self.to_ram:
            x = self.video[seq_i:seq_i + self.window_length + 1]
            y = torch.tensor(self.label[seq_i:seq_i + self.window_length], dtype=torch.float32)
        else:
            x = self.file['preprocessed_video'][seq_i:seq_i + self.window_length + 1]
            y = torch.tensor(self.file['preprocessed_label'][seq_i:seq_i + self.window_length], dtype=torch.float32)
        return x, y
