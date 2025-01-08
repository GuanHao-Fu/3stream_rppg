import torch
from nets.models.sub_models.AppearanceModel import AppearanceModel_2D
from nets.models.sub_models.LinearModel import LinearModel
from nets.models.sub_models.MotionModel import MotionModel_Hybrid
from torchvision import transforms


class STM_Phys_1(torch.nn.Module):
    def __init__(self, frame_depth, pop_mean, pop_std, eca, skip):
        super().__init__()
        self.transforms_app = transforms.Compose([
            transforms.Normalize(mean=pop_mean[0], std=pop_std[0]),
            transforms.RandomHorizontalFlip(0.),
            transforms.RandomVerticalFlip(0.)])

        self.transforms_motion = transforms.Compose([
            transforms.Normalize(mean=pop_mean[1], std=pop_std[1]),
            transforms.RandomHorizontalFlip(0.),
            transforms.RandomVerticalFlip(0.)])

        self.in_channels = 3
        self.out_channels = 32
        self.kernel_size = (3, 3)
        self.attention_mask1 = None
        self.attention_mask2 = None
        self.appearance_model = AppearanceModel_2D(eca, in_channels=self.in_channels, out_channels=self.out_channels,
                                                   kernel_size=self.kernel_size)

        self.motion_model = MotionModel_Hybrid(eca, self.in_channels,  self.out_channels, self.kernel_size,
                                               frame_depth, skip=skip)

        self.hr_linear_model = LinearModel(eca)

    def forward(self, x):
        # X has shape: B, window_length (T) + 1, H, W, 6 - last frame at T+1 th is used to calculate the motion map
        T = x.shape[1]
        x = x / 255
        x = torch.tensor_split(x, 2, dim=4)  # Split into motion and appearance stream
        appearance_input = x[1]
        appearance_input = appearance_input[:, :-1, :, :, :].permute(0, 1, 4, 2, 3)  # -> B, T, 3, H, W
        appearance_input = self.transforms_app(appearance_input)

        motion_input = x[0]  # -> B, T+1, H, W, 3
        for i in range(T-1):
            motion_input[:, i, :, :, :] = motion_input[:, i+1, :, :, :] - motion_input[:, i, :, :, :]
        motion_input = self.transforms_motion(motion_input[:, :-1, :, :, :].permute(0, 1, 4, 2, 3))

        self.attention_mask1, self.attention_mask2 = self.appearance_model(appearance_input)  #-> B*T, C, H, W
        motion_output = self.motion_model(motion_input, self.attention_mask1, self.attention_mask2)
        hr_out = self.hr_linear_model(motion_output)

        return hr_out

    def get_attention_mask(self):
        return self.attention_mask1, self.attention_mask2
