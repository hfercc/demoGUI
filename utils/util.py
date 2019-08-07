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

from sklearn import manifold
from sklearn.metrics import pairwise
from sklearn.utils import extmath


non_image_vars = ['Age', 'PTGENDER', 'PTEDUCAT', 'APOE Status', 'MMSCORE', 'CDR', 'AVLT-LTM', 'AVLT-Total', 'ADAS']
one_hot_vars = {"APOE Status": {'NC': 0, 'HT': 1, 'HM': 2, 0.0: 3}}
dx2label = {"AD": 0, "MCI": 1, "NL": 2}


def one_hot_torch(index, classes):
    '''
    index: labels, batch_size * 1, index starts from 0
    classes: int, # of classes
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
    return Variable(y_onehot).cuda()


def focal_loss(input, y, alpha=0.25, gamma=2, eps=1e-7, reduction='elementwise_mean', reverse_weighting=False):
    # print("focal loss:", input, target)
    y = y.view(-1, 1)
    y_hot = one_hot_torch(y, input.size(-1))
    logit = F.softmax(input, dim=-1)
    logit = logit.clamp(eps, 1. - eps)

    loss = -1 * y_hot * torch.log(logit) # cross entropy
    if reverse_weighting:
        for i in range(loss.size()[0]):
            index = torch.argmax(y_hot)
            loss[i, index] = loss[i, index] * (1 - logit[i, 1 - index]) ** gamma
        loss *= alpha
    else:
        loss = alpha * loss * (1 - logit) ** gamma # focal loss

    if reduction == 'elementwise_mean':
        return loss.sum() / input.size()[0]
    elif reduction == 'sum':
        return loss.sum()
    elif reduction == 'elementwise_sum':
        return loss.sum(dim=1)
    else:
        return loss


class FocalLoss(nn.Module):

    def __init__(self, alpha=0.25, gamma=2, eps=1e-7):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps

    def forward(self, input, y):
        return focal_loss(input, y, self.alpha, self.gamma, self.eps)


def sigmoid_loss(input, y, reduction='elementwise_mean'):
    # y must be -1/+1
    y = y.view(-1, 1)
    # NOTE: torch.nn.functional.sigmoid is 1 / (1 + exp(-x)). BUT sigmoid loss should be 1 / (1 + exp(x))
    loss = torch.nn.functional.sigmoid(-input * y)

    if reduction == 'elementwise_mean':
        return loss.sum() / input.size()[0]
    elif reduction == 'sum':
        return loss.sum()
    elif reduction == 'elementwise_sum':
        return loss.sum(dim=1)
    else:
        return loss

class SigmoidLoss(nn.Module):

    def __init__(self, reduction='elementwise_mean'):
        super(SigmoidLoss, self).__init__()
        self.reduction = reduction

    def forward(self, input, y):
        return sigmoid_loss(input, y, self.reduction)


def edge_weight(In_data): 
    Rho = 1e-2
    # https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/utils/extmath.py
    X = extmath.row_norms(In_data, squared=True) # Row-wise (squared) Euclidean norm of X.
    X = X[:,np.newaxis]
    kernel = np.dot(In_data, In_data.T)
    XX = np.ones((len(X), 1))
    X = np.dot(X, XX.T)
    kernel *= -2
    kernel = X + kernel + X.T
    kernel = np.exp(-Rho * kernel)
    return kernel


def laplacian(In_data, normal=False):
    In_data = In_data.reshape(len(In_data), -1)
    # In_data = np.float128(In_data)/255.
    adj_mat = edge_weight(In_data)
    D = np.zeros((len(In_data), len(In_data)))
    for n in range(len(D)):
        D[n,n] = np.sum(adj_mat[n,:])
    if normal == True:
        sqrt_deg_matrix = np.mat(np.diag(np.diag(D)**(-0.5)))
        lap_matrix = sqrt_deg_matrix * np.mat(D - adj_mat) * sqrt_deg_matrix
    else:
        lap_matrix = D - adj_mat
    return (np.float32(lap_matrix))

def pu_risk_estimators_sigmoid(y_pred, y_true):
    # y_true is -1/1
    one_u = torch.ones(y_true.size()).cuda()
    u_mask = torch.abs(y_true - one_u)
    
    P_size = torch.sum(y_true)
    u_size = torch.sum(u_mask)
    P_p = (sigmoid_loss(y_pred, one_u, reduction='elementwise_sum')).dot(y_true) / P_size if P_size > 0 else 0 # should go down
    P_n = (sigmoid_loss(y_pred, - one_u, reduction='elementwise_sum')).dot(y_true) / P_size if P_size > 0 else 0 # should go up
    P_u = (sigmoid_loss(y_pred, - one_u, reduction='elementwise_sum')).dot(u_mask) / u_size if u_size > 0 else 0 # should go down
    return P_p, P_n, P_u

def pu_risk_estimators_focal(y_pred, y_true):
    # y_pred is [score1, score2] before softmax logit, y_true is 0/1
    one_u = torch.ones(y_true.size()).cuda()
    zeros = torch.zeros(y_true.size()).cuda()
    u_mask = torch.abs(y_true - one_u)
    
    P_size = torch.max(torch.sum(y_true), torch.Tensor([1]).cuda())
    u_size = torch.max(torch.sum(u_mask), torch.Tensor([1]).cuda())
    P_p = (focal_loss(y_pred, one_u, gamma=3, reduction='elementwise_sum')).dot(y_true) / P_size # should go down
    P_n = (focal_loss(y_pred, zeros, gamma=3, reduction='elementwise_sum')).dot(y_true) / P_size # should go up
    P_u = (focal_loss(y_pred, zeros, gamma=3, reduction='elementwise_sum')).dot(u_mask) / u_size # should go down
    return P_p, P_n, P_u


def pu_loss(y_pred, y_true, loss_fn, Probility_P=0.25, BETA=0, gamma=1.0, Yi=1e-8, L=None):
    P_p, P_n, P_u = 0, 0, 0
    if loss_fn == "sigmoid":
        P_p, P_n, P_u = pu_risk_estimators_sigmoid(y_pred, y_true)
    elif loss_fn == "focal":
        P_p, P_n, P_u = pu_risk_estimators_focal(y_pred, y_true)
    else: pass

    M_reg = torch.zeros(1)
    if L is not None:
        FL = torch.mm((2 * y_pred - 1).transpose(0, 1), L)
        R_manifold = torch.mm(FL, (2 * y_pred - 1))
        M_reg = Yi * R_manifold
    
    PU_1 = Probility_P * P_p + P_u - Probility_P * P_n
    PU_2 = P_u - Probility_P * P_n
    if -BETA > PU_2:
        return -gamma * PU_2 + torch.sum(M_reg), torch.sum(M_reg)#, Probility_P * P_p, P_u, Probility_P * P_n
        # return -gamma * PU_2, torch.sum(M_reg), Probility_P * P_p, P_u, Probility_P * P_n
        # return Probility_P * P_p
    else:
        return PU_1 + torch.sum(M_reg), torch.sum(M_reg)#, Probility_P * P_p, P_u, Probility_P * P_n
        # return PU_1, torch.sum(M_reg), Probility_P * P_p, P_u, Probility_P * P_n


class PULoss(nn.Module):
    '''
    only works for binary classification
    '''

    def __init__(self, loss_fn='sigmoid', Probility_P=0.25, BETA=0, gamma=1.0, Yi=1e-8):
        super(PULoss, self).__init__()
        self.loss_fn = loss_fn
        self.Probility_P = Probility_P
        self.BETA = BETA
        self.gamma = gamma
        self.Yi = Yi

    def forward(self, y_pred, y_true, L=None):
        return pu_loss(y_pred, y_true, self.loss_fn, self.Probility_P, self.BETA, self.gamma, self.Yi, L)



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

