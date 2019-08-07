import torch
from torch import nn
import torch.nn.functional as F

'''
Esmaeilzadeh, Soheil, Dimitrios Ioannis Belivanis, Kilian M. Pohl, and Ehsan Adeli. "End-To-End Alzheimer’s Disease Diagnosis and Biomarker Identification." In International Workshop on Machine Learning in Medical Imaging, pp. 337-345. Springer, Cham, 2018.

DX classification
'''
class Soheil(nn.Module):
    def __init__(self):
        super(Soheil, self).__init__()

        self.conv = nn.Sequential(
            # 121, 145, 121
            # padding tuple (padT, padH, padW)
            nn.Conv3d(1, 16, kernel_size=3, stride=1, padding=1), # b, 8, 121, 145, 121
            nn.BatchNorm3d(16),
            nn.ReLU(True),
            nn.MaxPool3d(kernel_size=2, stride=2, padding=1), # b, 16, 61, 73, 61
            nn.Conv3d(16, 32, kernel_size=3, stride=1, padding=1), # b, 16, 61, 73, 61
            nn.BatchNorm3d(32),
            nn.ReLU(True),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1),  # b, 16, 31, 37, 31
            nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1),  # b, 32, 31, 37, 31
            nn.BatchNorm3d(64),
            nn.ReLU(True),
            nn.MaxPool3d(kernel_size=4, stride=2, padding=1),  # b, 32, 16, 19, 16
        )
        nn.init.xavier_uniform(self.conv[0].weight)
        nn.init.xavier_uniform(self.conv[4].weight)
        nn.init.xavier_uniform(self.conv[8].weight)
        self.fc = nn.Sequential(
            nn.Linear(64 * 13 * 16 * 14 + 2, 256),
            nn.BatchNorm1d(256),
            nn.Dropout1d(p=0.4),
            nn.ReLU(True),
            nn.Linear(256, 2),
            nn.ReLU(True)
        )
        nn.init.xavier_uniform(self.fc[0].weight)
        nn.init.xavier_uniform(self.fc[3].weight)
        nn.init.xavier_uniform(self.fc[6].weight)

    def forward(self, x, ages, genders):
        x = self.conv(x)
        # print(x.size())
        x = x.view(-1, 128 * 13 * 16 * 14)
        x = torch.cat((x, ages, genders), dim=1)
        x = self.fc(x)
        return x
