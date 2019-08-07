import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from collections import Counter
import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
from skimage import transform


non_image_vars = ['Age', 'PTGENDER', 'PTEDUCAT', 'APOE Status', 'MMSCORE', 'CDR', 'AVLT-LTM', 'AVLT-Total', 'ADAS']
one_hot_vars = {"APOE Status": {'NC': 0, 'HT': 1, 'HM': 2, 0.0: 3}}
dx2label = {"AD": 0, "MCI": 1, "NL": 2}


def one_hot_torch(index, classes, cuda):
    '''
    index: labels, batch_size * 1, index starts from 0
    classes: int, # of classes
    cuda: boolean
    '''
    y = index.type(torch.LongTensor)
    # One hot encoding buffer that you create out of the loop and just keep reusing
    y_onehot = torch.FloatTensor(y.size()[0], classes)
    y_onehot.zero_()
    '''
        TypeError: scatter_ received an invalid combination of arguments - got (int, Variable, int), but expected one of:
     * (int dim, torch.LongTensor index, float value)
          didn't match because some of the arguments have invalid types: (int, Variable, int)
     * (int dim, torch.LongTensor index, torch.FloatTensor src)
          didn't match because some of the arguments have invalid types: (int, Variable, int)
      '''
    y_onehot.scatter_(1, y.data, 1)
    if cuda: return Variable(y_onehot).cuda()
    else: return Variable(y_onehot)


class FocalLoss(nn.Module):

    def __init__(self, alpha=0.25, gamma=2, eps=1e-7):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps

    def forward(self, input, target, cuda):
        # print("focal loss:", input, target)
        y = one_hot_torch(target, input.size(-1), cuda)
        logit = F.softmax(input, dim=-1)
        logit = logit.clamp(self.eps, 1. - self.eps)

        loss = -1 * y * torch.log(logit) # cross entropy
        loss = self.alpha * loss * (1 - logit) ** self.gamma # focal loss

        return loss.sum() / input.size()[0]


def L1_reg(model):
    # https://discuss.pytorch.org/t/how-does-one-implement-weight-regularization-l1-or-l2-manually-without-optimum/7951
    l1_reg = None
    for W in model.parameters():
        if l1_reg is None:
            l1_reg = W.norm(1)
        else:
            l1_reg = l1_reg + W.norm(1)
    return l1_reg


def suvr2class(suvrs):
    labels = torch.round((suvrs - 0.8) * 10).type(torch.LongTensor)
    return labels


def get_batch(rids, df, data_path, prefix, image = True, non_image = False, mask = None):
    if image: images = []
    suvrs = []
    if non_image: non_image_data = []
    for id in rids:
        # if image: images.append(np.load(os.path.join(data_path, prefix + str(id) + ".npy"))[8:112, 10:138, :112])
        if mask is not None: images.append(np.load(os.path.join(data_path, prefix + str(id) + ".npy")) * mask)
        elif image: images.append(np.load(os.path.join(data_path, prefix + str(id) + ".npy")))
        suvrs.append(df.loc[df["RID"] == id]["Reference Region: avidcereb"].tolist()[0])
        if non_image:
            non_image_row = []
            for v in non_image_vars:
                if v in one_hot_vars:
                    tmp = [0] * len(one_hot_vars[v].keys())
                    tmp[one_hot_vars[v][df.loc[df["RID"] == id][v].tolist()[0]]] = 1
                    non_image_row += tmp
                else:
                    non_image_row.append(df.loc[df["RID"] == id][v].tolist()[0])
            non_image_data.append(non_image_row)
    if image:
        if non_image: return images, suvrs, non_image_data, len(non_image_data[0])
        else: return images, suvrs
    else:
        if non_image: return suvrs, non_image_data, len(non_image_data[0])
        else: return suvrs



def get_batch_pet(rids, df, data_path, prefix):
    # images = []
    images = np.empty((len(rids), 44, 44, 44))
    suvrs = []
    for i in range(len(rids)):
        id = rids[i]
        ##########################
        # images.append(np.load(os.path.join(data_path, prefix + str(id) + ".npy"))[24: -24, 30: -30, 24: -24])
        ##########################
        tmp = np.load(os.path.join(data_path, prefix + str(id) + ".npy"))
        ## origin rescaled & normalized pet ########################
        # tmp /= 255.
        # tmp = transform.resize(tmp, (64, 64, 64))
        # tmp[tmp < 0.2] = 0
        # tmp[tmp > 0.5] = 0.75
        # tmp[(tmp >= 0.2) & (tmp <= 0.5)] = 0.35
        ##########################
        # tmp /= 255.
        # tmp = transform.resize(tmp, (64, 64, 64))
        # tmp[tmp <= 0.46] = 0
        # tmp[tmp > 0.46] = 0.75
        ##########################
        images[i, :, :, :] = tmp[10:54, 10:54, 10:54]
        suvrs.append(df.loc[df["RID"] == id]["Reference Region: avidcereb"].tolist()[0])
    return images, suvrs



def get_patch(rids, df, data_path, prefix, xmin, xmax, ymin, ymax, zmin, zmax):
    images = []
    for id in rids:
        if prefix.startswith("MRI"):
            # images.append(np.load(os.path.join(data_path, prefix + str(id) + ".npy")))
            tmp = np.pad(np.load(os.path.join(data_path, prefix + str(id) + ".npy"))[8:112, 10:138, :112], ((27, 27), (27, 27), (27, 27)), 'constant', constant_values=0)
            images.append(tmp[xmin:xmax, ymin:ymax, zmin:zmax])
        else:
            images.append(np.load(os.path.join(data_path, prefix + str(id) + ".npy"))[xmin:xmax, ymin:ymax, zmin:zmax])
    return images



def get_patch_all(rids, data_path, w_mri, w_pet, s1, s2, s3):
    mris, pets = np.empty((10 * 12 * 10 * len(rids), w_mri, w_mri, w_mri)), np.empty((10 * 12 * 10 * len(rids), w_pet, w_pet, w_pet))
    for i in range(len(rids)):
        id = rids[i]
        mri = np.pad(np.load(os.path.join(data_path, "MRI_norm_grey_" + str(id) + ".npy"))[8:112, 10:138, :112], ((6, 6), (6, 6), (6, 6)), 'constant', constant_values=0)
        pet = np.load(os.path.join(data_path, "AV45_norm_" + str(id) + ".npy"))
        for x in range(10):
            for y in range(12):
                for z in range(10):
                    mris[i * 10 * 12 * 10 + x * 12 * 10 + y * 10 + z, :, :, :] = mri[round(x * s1): round(x * s1) + 24, round(y * s2): round(y * s2) + 24, round(z * s3): round(z * s3) + 24]
                    pets[i * 10 * 12 * 10 + x * 12 * 10 + y * 10 + z, :, :, :] = pet[x * w_pet: (x + 1) * w_pet, y * w_pet: (y + 1) * w_pet, z * w_pet: (z + 1) * w_pet]
    return mris, pets



def get_patch_all_resized_normalized(rids, data_path, w_mri = 16, w_pet = 4, s1 = 4, s2 = 4, s3 = 4):
    # 64 * 64 * 64
    mris, pets = np.empty((16 * 16 * 16 * len(rids), w_mri, w_mri, w_mri)), np.empty((16 * 16 * 16 * len(rids), w_pet, w_pet, w_pet))
    for i in range(len(rids)):
        id = rids[i]
        
        mri = np.load(os.path.join(data_path, "MRI_norm_grey_" + str(id) + ".npy"))[8:112, 10:138, :112]
        mri = transform.resize(mri, (64, 64, 64))
        mri /= 255.
        mri = np.pad(mri, ((6, 6), (6, 6), (6, 6)), 'constant', constant_values=0)
        
        pet = np.load(os.path.join(data_path, "AV45_norm_" + str(id) + ".npy"))
        pet /= 255.
        pet = transform.resize(pet, (64, 64, 64))
        # pet[pet < 0.2] = 0
        # pet[pet > 0.5] = 2 # 0.75
        # pet[(pet >= 0.2) & (pet <= 0.5)] = 1 # 0.35
        pet[pet <= 0.46] = 0
        pet[pet > 0.46] = 1 # 0.75
        
        for x in range(16):
            for y in range(16):
                for z in range(16):
                    mris[i * 16 * 16 * 16 + x * 16 * 16 + y * 16 + z, :, :, :] = mri[round(x * s1): round(x * s1) + 16, round(y * s2): round(y * s2) + 16, round(z * s3): round(z * s3) + 16]
                    pets[i * 16 * 16 * 16 + x * 16 * 16 + y * 16 + z, :, :, :] = pet[x * w_pet: (x + 1) * w_pet, y * w_pet: (y + 1) * w_pet, z * w_pet: (z + 1) * w_pet]
    return mris, pets



def get_patch_part_resized_normalized(rids, data_path, w_mri = 16, w_pet = 4, s1 = 4, s2 = 4, s3 = 4):
    # 44 * 44 * 44
    mris, pets = np.empty((11 * 11 * 11 * len(rids), w_mri, w_mri, w_mri)), np.empty((11 * 11 * 11 * len(rids), w_pet, w_pet, w_pet))
    for i in range(len(rids)):
        id = rids[i]
        
        mri = np.load(os.path.join(data_path, "MRI_norm_grey_" + str(id) + ".npy"))[8:112, 10:138, :112]
        mri = transform.resize(mri, (64, 64, 64))[10:54, 10:54, 10:54]
        mri /= 255.
        mri = np.pad(mri, ((6, 6), (6, 6), (6, 6)), 'constant', constant_values=0)
        
        pet = np.load(os.path.join(data_path, "AV45_norm_" + str(id) + ".npy"))
        pet /= 255.
        pet = transform.resize(pet, (64, 64, 64))[10:54, 10:54, 10:54]
        # pet[pet < 0.2] = 0
        # pet[pet > 0.5] = 2 # 0.75
        # pet[(pet >= 0.2) & (pet <= 0.5)] = 1 # 0.35
        pet[pet <= 0.46] = 0
        pet[pet > 0.46] = 1 # 0.75
        
        for x in range(11):
            for y in range(11):
                for z in range(11):
                    mris[i * 11 * 11 * 11 + x * 11 * 11 + y * 11 + z, :, :, :] = mri[round(x * s1): round(x * s1) + w_mri, round(y * s2): round(y * s2) + w_mri, round(z * s3): round(z * s3) + w_mri]
                    pets[i * 11 * 11 * 11 + x * 11 * 11 + y * 11 + z, :, :, :] = pet[x * w_pet: (x + 1) * w_pet, y * w_pet: (y + 1) * w_pet, z * w_pet: (z + 1) * w_pet]
    return mris, pets




def get_patch_part(rids, data_path, w_mri, w_pet, s1, s2, s3):
    mris, pets = np.empty((6 * 8 * 6 * len(rids), w_mri, w_mri, w_mri)), np.empty((6 * 8 * 6 * len(rids), w_pet, w_pet, w_pet))
    for i in range(len(rids)):
        id = rids[i]
        mri = np.pad(np.load(os.path.join(data_path, "MRI_norm_grey_" + str(id) + ".npy"))[8:112, 10:138, :112], ((6, 6), (6, 6), (6, 6)), 'constant', constant_values=0)
        pet = np.load(os.path.join(data_path, "AV45_norm_" + str(id) + ".npy"))
        for x in range(6):
            for y in range(8):
                for z in range(6):
                    mris[i * 6 * 8 * 6 + x * 8 * 6 + y * 6 + z, :, :, :] = mri[round((x + 2) * s1): round((x + 2) * s1) + 24, round((y + 2) * s2): round((y + 2) * s2) + 24, round((z + 2) * s3): round((z + 2) * s3) + 24]
                    pets[i * 6 * 8 * 6 + x * 8 * 6 + y * 6 + z, :, :, :] = pet[(x + 2) * w_pet: ((x + 2) + 1) * w_pet, (y + 2) * w_pet: ((y + 2) + 1) * w_pet, (z + 2) * w_pet: ((z + 2) + 1) * w_pet]
    return mris, pets



def get_2d(rids, df, data_path, prefix, zmin, zmax, n):
    if n == -1: zs = range(zmin, zmax)
    else: zs = np.random.choice(range(zmin, zmax), n, replace = False)
    images = np.empty((len(rids) * len(zs), 104, 128))
    labels = [""] * len(rids) * len(zs)
    for i in range(len(rids)):
        id = rids[i]
        # images.append(np.load(os.path.join(data_path, prefix + str(id) + ".npy")))
        tmp = np.load(os.path.join(data_path, prefix + str(id) + ".npy"))[8:112, 10:138, :]
        for j in range(len(zs)):
            z = zs[j]
            images[i * len(zs) + j, :, :] = tmp[:, :, z]
            labels[i * len(zs) + j] = df.loc[df["RID"] == id]["DX"].tolist()[0]
    return images, labels


def show_slices(slices, lower = None, upper = None):
    fig, axes = plt.subplots(1, len(slices), figsize=(30,30))
    for i, slice in enumerate(slices):
        if lower != None and upper != None: axes[i].imshow(slice.T, cmap="gray", origin="lower", vmin=lower, vmax=upper)
        elif lower != None: axes[i].imshow(slice.T, cmap="gray", origin="lower", vmin=lower)
        elif upper != None: axes[i].imshow(slice.T, cmap="gray", origin="lower", vmax=upper)
        else: axes[i].imshow(slice.T, cmap="gray", origin="lower")



def confusion_matrix(predictions, truths, classes):
    '''
    predictions, truths: list of integers
    classes: int, # of classes
    return confusion_matrix: x-axis target, y-axis predictions
    '''
    m = np.zeros((classes, classes))
    accuracy = np.zeros(classes)
    for i in range(len(predictions)):
        m[int(predictions[i]), int(truths[i])] += 1
    diagonal = 0
    for i in range(classes):
        accuracy[i] = m[i, i] / np.sum(m[:, i], axis=0)
        diagonal += m[i, i]
    return m, accuracy, float(diagonal) / len(predictions)

