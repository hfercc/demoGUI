import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

'''
ad & subject_id classification
'''
class _Basic2Conv(nn.Module):
    def __init__(self):
        super(_Basic2Conv, self).__init__()
        self.output_dim = 32 * 26 * 32 * 28
        self.conv = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(True),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1),
            nn.Conv3d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(True),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1),
        )
        nn.init.xavier_uniform(self.conv[0].weight)
        nn.init.xavier_uniform(self.conv[4].weight)

    def forward(self, x):
        b, c, d, h, w = x.size()
        return self.conv(x).view(b, -1)


class _Basic3Conv(nn.Module):
    def __init__(self):
        super(_Basic3Conv, self).__init__()
        self.output_dim = 64 * 13 * 16 * 14
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
        b, c, d, h, w = x.size()
        return self.conv(x).view(b, -1)


class _Basic4Conv(nn.Module):
    def __init__(self):
        super(_Basic4Conv, self).__init__()
        self.output_dim = 64 * 7 * 8 * 7
        self.conv = nn.Sequential(
            nn.Conv3d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(8),
            nn.ReLU(True),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1),
            nn.Conv3d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(True),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1),
            nn.Conv3d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(True),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1),
            nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(True),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1),
        )
        nn.init.xavier_uniform(self.conv[0].weight)
        nn.init.xavier_uniform(self.conv[4].weight)
        nn.init.xavier_uniform(self.conv[8].weight)
        nn.init.xavier_uniform(self.conv[12].weight)

    def forward(self, x):
        b, c, d, h, w = x.size()
        return self.conv(x).view(b, -1)


class _Global_Hippo_2Conv(nn.Module):
    def __init__(self):
        super(_Global_Hippo_2Conv, self).__init__()
        self.output_dim = 32 * (13**3 + 2 * 8**3)
        self.conv_mri = _Basic2Conv()
        self.conv_left = _Basic2Conv()
        self.conv_right = _Basic2Conv()
        # self.output_dim = 64 * (6**3 + 2 * 4**3)
        # self.conv_mri = _Basic3Conv()
        # self.conv_left = _Basic3Conv()
        # self.conv_right = _Basic3Conv()
    def forward(self, mri, left, right):
        '''
        mri: 50x50x50
        left/right: 30x30x30
        '''
        mri = self.conv_mri(mri)
        left = self.conv_left(left)
        right = self.conv_right(right)
        return torch.cat((mri, left, right), dim=1)


class _Classifier_AD(nn.Module):
    def __init__(self, dim, n_class):
        super(_Classifier_AD, self).__init__()
        self.input_dim = dim
        self.n_class = n_class
        self.fc = nn.Sequential(
            nn.Linear(self.input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.Linear(256, self.n_class),
            nn.ReLU(True)
        )
        nn.init.xavier_uniform(self.fc[0].weight)
        nn.init.xavier_uniform(self.fc[3].weight)

    def forward(self, x):
        # x = x.view(-1, self.dim)
        return self.fc(x)


class _Classifier_AD_Demo(nn.Module):
    def __init__(self, dim, n_class):
        super(_Classifier_AD_Demo, self).__init__()
        self.input_dim = dim
        self.n_class = n_class

        self.fc1 = nn.Sequential(
            nn.Linear(self.input_dim, 512),
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
            nn.Linear(64, self.n_class),
            nn.ReLU(True)
        )
        nn.init.xavier_uniform(self.fc2[0].weight)

    def forward(self, x, ages, genders, edus, apoes):
        x = self.fc1(x)
        x = torch.cat((x, ages, genders, edus, apoes), dim=1)
        return self.fc2(x)


class _Classifier_RID(nn.Module):
    def __init__(self, dim, n_class):
        super(_Classifier_RID, self).__init__()
        self.input_dim = dim
        self.n_class = n_class
        self.fc = nn.Sequential(
            nn.Linear(self.input_dim, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(True),
            nn.Linear(2048, self.n_class),
            nn.ReLU(True)
        )
        nn.init.xavier_uniform(self.fc[0].weight)
        nn.init.xavier_uniform(self.fc[3].weight)

    def forward(self, x):
        # x = x.view(-1, self.dim)
        return self.fc(x)


class Classifier(nn.Module):
    def __init__(self, n_class_pos, n_class_neg):
        super(Classifier, self).__init__()
        self.n_class_pos = n_class_pos
        self.n_class_neg = n_class_neg

        # self.feature_extractor = _Basic4Conv()
        # self.feature_extractor = _Basic2Conv()
        self.feature_extractor = _Global_Hippo_2Conv()
        self.flatten_dim = self.feature_extractor.output_dim
        # self.classifier_pos = _Classifier_AD(self.flatten_dim, self.n_class_pos)
        self.classifier_pos = _Classifier_AD_Demo(self.flatten_dim, self.n_class_pos)
        self.classifier_neg = _Classifier_RID(self.flatten_dim, self.n_class_neg)

    # def forward(self, mri):
    #     mri = self.feature_extractor(mri)
    #     # print(mri.size())
    #     # mri = mri.view(-1, self.dim)
    #     y_pos = self.classifier_pos(mri)
    #     y_neg = self.classifier_neg(mri)
    #     return y_pos, y_neg
    def forward(self, mri, left, right, ages, genders, edus, apoes):
        x = self.feature_extractor(mri, left, right)
        y_pos = self.classifier_pos(x, ages, genders, edus, apoes)
        y_neg = self.classifier_neg(x)
        return y_pos, y_neg
