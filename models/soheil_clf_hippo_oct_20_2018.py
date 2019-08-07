import torch
from torch import nn
import torch.nn.functional as F

'''
Esmaeilzadeh, Soheil, Dimitrios Ioannis Belivanis, Kilian M. Pohl, and Ehsan Adeli. "End-To-End Alzheimer’s Disease Diagnosis and Biomarker Identification." In International Workshop on Machine Learning in Medical Imaging, pp. 337-345. Springer, Cham, 2018.

DX classification
'''

class Soheil_Conv(nn.Module):
    def __init__(self):
        super(Soheil_Conv, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(True),
            nn.MaxPool3d(kernel_size=2, stride=2, padding=1),
            nn.Conv3d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(True),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1),
            nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(True),
            nn.MaxPool3d(kernel_size=4, stride=2, padding=1),
        )
        nn.init.xavier_uniform(self.conv[0].weight)
        nn.init.xavier_uniform(self.conv[4].weight)
        nn.init.xavier_uniform(self.conv[8].weight)

    def forward(self, x):
        return self.conv(x)


class Soheil(nn.Module):
    def __init__(self):
        super(Soheil, self).__init__()

        self.conv_mri = Soheil_Conv()
        self.conv_left = Soheil_Conv()
        self.conv_right = Soheil_Conv()

        # self.fc = nn.Sequential(
        #     nn.Linear(64 * (13*13*13 + 2*8*8*8) + 2, 512),
        #     nn.BatchNorm1d(512),
        #     nn.Dropout1d(p=0.4),
        #     nn.ReLU(True),
        #     nn.Linear(512, 2),
        #     nn.ReLU(True)
        # )
        # nn.init.xavier_uniform(self.fc[0].weight)
        # nn.init.xavier_uniform(self.fc[3].weight)
        # nn.init.xavier_uniform(self.fc[6].weight)

        self.fc1 = nn.Sequential(
            nn.Linear(64 * (6**3 + 2*4**3), 512),
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
            nn.Linear(64, 2),
            nn.ReLU(True)
        )
        nn.init.xavier_uniform(self.fc2[0].weight)

    # def forward(self, mri, left, right, ages, genders):
    def forward(self, mri, left, right, ages, genders, edus, apoes):
        mri = self.conv_mri(mri)
        left = self.conv_mri(left)
        right = self.conv_mri(right)
        # print(mri.size(), left.size(), right.size())
        mri = mri.view(-1, 64*6**3)
        left = left.view(-1, 64*4**3)
        right = right.view(-1, 64*4**3)
        # x = torch.cat((mri, left, right, ages, genders), dim=1)
        # x = self.fc(x)

        x = torch.cat((mri, left, right), dim=1)
        x = self.fc1(x)
        x = torch.cat((x, ages, genders, edus, apoes), dim=1)
        x = self.fc2(x)

        return x
