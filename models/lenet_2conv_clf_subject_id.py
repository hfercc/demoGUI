import torch
from torch import nn
import torch.nn.functional as F

'''
subject_id classification
'''
class Basic2Conv(nn.Module):
    def __init__(self):
        super(Basic2Conv, self).__init__()
        self.conv = nn.Sequential(
            # 121, 145, 121
            # padding tuple (padT, padH, padW)
            nn.Conv3d(1, 8, kernel_size=3, stride=1, padding=1), # b, 16, 61, 73, 61
            nn.BatchNorm3d(8),
            nn.ReLU(True),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1),  # b, 16, 31, 37, 31
            nn.Conv3d(8, 16, kernel_size=3, stride=1, padding=1),  # b, 32, 31, 37, 31
            nn.BatchNorm3d(16),
            nn.ReLU(True),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1),  # b, 32, 16, 19, 16
            nn.Conv3d(16, 32, kernel_size=3, stride=1, padding=1),  # b, 32, 31, 37, 31
            nn.BatchNorm3d(32),
            nn.ReLU(True),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1),  # b, 32, 16, 19, 16
            nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1),  # b, 32, 31, 37, 31
            nn.BatchNorm3d(64),
            nn.ReLU(True),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1),  # b, 32, 16, 19, 16
        )
        nn.init.xavier_uniform(self.conv[0].weight)
        nn.init.xavier_uniform(self.conv[4].weight)
        nn.init.xavier_uniform(self.conv[8].weight)
        nn.init.xavier_uniform(self.conv[12].weight)

    def forward(self, x):
        return self.conv(x)



class Lenet3D(nn.Module):
    def __init__(self):
        super(Lenet3D, self).__init__()
        self.conv_mri = Basic2Conv()
        self.fc = nn.Sequential(
            nn.Linear(64 * 7 * 8 * 7, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Linear(1024, 644),
            nn.ReLU(True)
        )
        nn.init.xavier_uniform(self.fc[0].weight)

    def forward(self, mri):
        mri = self.conv_mri(mri)
        # print(mri.size())
        mri = mri.view(-1, 64 * 7 * 8 * 7)
        return self.fc(mri)
