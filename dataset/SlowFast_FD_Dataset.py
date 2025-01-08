import torch
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm
from torchvision import transforms
# from utils.funcs import BPF_signal


class SlowFast_FD_Dataset(Dataset):
    def __init__(self, file, ds_name, window_length, valid):
        self.transform = torch.nn.Sequential(
            transforms.Resize((36, 36))
        )
        super(SlowFast_FD_Dataset, self).__init__()
        self.ds_name = ds_name
        self.size = 36
        self.valid = valid
        self.window_length = window_length
        self._get_arrays(file)

    def __len__(self):
        return self.total_length

    def __getitem__(self, idx):
        idx = int(idx * self.window_length)
        x = torch.tensor(self.video[idx:idx + self.window_length + 1], dtype=torch.float32).permute(0, 3, 1, 2)
        y = torch.tensor(self.label[idx:idx + self.window_length], dtype=torch.float32)
        x = self.transform(x)
        motion_fast = torch.empty((self.window_length, 3, self.size, self.size), dtype=torch.float32)
        motion_slow = torch.empty((self.window_length//2, 3, self.size, self.size), dtype=torch.float32)
        for i in range(self.window_length):
            motion_fast[i] = x[i+1] - x[i]
            if i % 2 == 0:
                j = i//2
                motion_slow[j] = x[i+2] - x[i]
        appearance_frame = x[:-1]
        return (motion_fast, motion_slow, appearance_frame), y

    def _get_arrays(self, file):
        with tqdm(total=len(list(file.keys())), position=0, leave=True, desc='Reading from file') as pbar:
            self.n_frames_per_video = np.empty((len(list(file.keys()))), dtype=int)
            for i, data_path in enumerate(list(file.keys())):
                n_frames_per_video = len(file[data_path]['label'])
                # if self.ds_name == 'MANHOB_HCI':
                #     self.n_frames_per_video[i] = n_frames_per_video // 2
                # else:
                self.n_frames_per_video[i] = n_frames_per_video
                video_frames = file[data_path]['video']
                labels = file[data_path]['label']
                if i == 0:
                    # if self.ds_name == 'MANHOB_HCI':
                    #     self.video = video_frames[::2]
                    #     self.label = labels[::2]
                    # else:
                        self.video = video_frames
                        self.label = labels
                else:
                    # if self.ds_name == 'MANHOB_HCI':
                    #     self.video = np.append(self.video, video_frames[::2], axis=0)
                    #     self.label = np.append(self.label, labels[::2])
                    # else:
                        self.video = np.append(self.video, video_frames, axis=0)
                        self.label = np.append(self.label, labels)
                pbar.update(1)
        self.total_length = (len(self.label) - 1) // self.window_length

    def update_state(self):
        self.pre_train = False
        self.total_length = (self.total_length * self.window_length) - self.window_length

