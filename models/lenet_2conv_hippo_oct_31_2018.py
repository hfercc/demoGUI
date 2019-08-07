import torch
from torch import nn
import torch.nn.functional as F

'''
SUVR regression image only: 3conv + 2fc
'''

class ConvBlock(nn.Module):
    def __init__(self):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=3, stride=1, padding=1), # b, 16, 61, 73, 61
            nn.BatchNorm3d(16),
            nn.ReLU(True),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1),  # b, 16, 31, 37, 31
            nn.Conv3d(16, 32, kernel_size=3, stride=1, padding=1),  # b, 32, 31, 37, 31
            nn.BatchNorm3d(32),
            nn.ReLU(True),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1),  # b, 32, 16, 19, 16
        )
        nn.init.xavier_uniform(self.conv[0].weight)
        nn.init.xavier_uniform(self.conv[4].weight)
    def forward(self, x):
        return self.conv(x)

class Lenet3D(nn.Module):
    def __init__(self):
        super(Lenet3D, self).__init__()

        self.conv_mri = ConvBlock()
        self.conv_left = ConvBlock()
        self.conv_right = ConvBlock()

        self.fc1 = nn.Sequential(
            nn.Linear(32 * (13**3 + 2*8**3), 512),
            nn.BatchNorm1d(512),
            # nn.Dropout(p=0.4),
            nn.ReLU(True),
            nn.Linear(512, 60),
            nn.BatchNorm1d(60),
            nn.ReLU(True)
        )
        nn.init.xavier_uniform(self.fc1[0].weight)
        nn.init.xavier_uniform(self.fc1[3].weight)
        self.fc2 = nn.Sequential(
            nn.Linear(64, 1),
            nn.ReLU(True)
        )
        nn.init.xavier_uniform(self.fc2[0].weight)

    def forward(self, mri, left, right, ages, genders, edus, apoes):
        mri = self.conv_mri(mri)
        left = self.conv_mri(left)
        right = self.conv_mri(right)
        # print(mri.size(), left.size(), right.size())
        mri = mri.view(-1, 32 * 13**3)
        left = left.view(-1, 32 * 8**3)
        right = right.view(-1, 32 * 8**3)
        x = torch.cat((mri, left, right), dim=1)
        x = self.fc1(x)
        x = torch.cat((x, ages, genders, edus, apoes), dim=1)
        x = self.fc2(x)
        return x
