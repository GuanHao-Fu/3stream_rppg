import torch
from nets.models.sub_models.AppearanceModel import AppearanceModel_STM
from nets.models.sub_models.LinearModel import LinearModel_STM
from nets.models.sub_models.MotionModel import MotionModel_STM
from torchvision import transforms


class STM_Phys(torch.nn.Module):
    def __init__(self, in_planes, pop_mean, pop_std):  # input: Batch, 3, 64, 64
        super().__init__()
        self.transforms = transforms.Compose([
            transforms.Normalize(mean=pop_mean, std=pop_std),
            transforms.RandomHorizontalFlip(0.),
            transforms.RandomVerticalFlip(0.)])

        self.motion_model = MotionModel_STM(in_planes)
        self.appearance_model = AppearanceModel_STM(False, 3, in_planes, kernel_size=3)
        self.hr_linear_model = LinearModel_STM()

    def forward(self, x):
        # X must have shape B, T+1, C, H, W where +1 at dim T is the last image.
        inputs = x / 255
        B, T, H, W, C = inputs.shape
        inputs = inputs.view(B*T, H, W, C)
        inputs = self.transforms(inputs.permute(0, 3, 1, 2))
        inputs = inputs.view(B, T, C, H, W)

        if torch.isnan(inputs).any():
            print('Input has nan')

        self.attention_mask1, self.attention_mask2 = self.appearance_model(inputs)
        motion_output = self.motion_model(inputs, self.attention_mask1, self.attention_mask2)
        hr_out = self.hr_linear_model(motion_output)

        return hr_out

