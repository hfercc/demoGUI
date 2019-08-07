import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
import os
from tensorboardX import SummaryWriter
from tqdm import tqdm
import sys

# task_name as name of logging file ###########
task_name = "clf_ad1nl0_mri50_hippo30_lrflip_lenet_3.17.2019.focal5.lr1e6"
print(task_name)

import sys
user = "chenwy"
# add system paths for models, datasets, utils ########################
sys.path.append("/home/chenwy/GV/suvr/github/")
sys.path.append("/home/chenwy/GV/suvr/github/models")
sys.path.append("/home/chenwy/GV/suvr/github/datasets")
sys.path.append("/home/chenwy/GV/suvr/github/utils")

from lenet_2conv_clf_oct_17_2018 import Lenet3D as Model
from adni_dataset import ADNI
from util import FocalLoss, focal_loss, SigmoidLoss, laplacian
from metrics import ConfusionMatrix

# where MRI images locate ######################
if user == "chenwy": data_path = "/ssd1/chenwy/adni/"
else: data_path = "/data3/temp_data/gvwuyangchen/"

# where to save pytorch .pth model weight file #####################
if user == "chenwy": model_path = "/home/chenwy/GV/suvr/github/jobs/saved_models/"
else: model_path = "/home/gvwuyangchen/PUlearning/saved_models/"
if not os.path.isdir(model_path): os.mkdir(model_path)

# where to save log files #############################
if user == "chenwy": log_path = "/home/chenwy/GV/suvr/github/jobs/runs/"
else: log_path = "/home/gvwuyangchen/PUlearning/runs/"
if not os.path.isdir(log_path): os.mkdir(log_path)

# load list of names of MRI images ############################
ids_train = np.load(os.path.join(data_path, "rid.image_id.train.adni.npy"))
ids_val = np.load(os.path.join(data_path, "rid.image_id.test.adni.npy"))
# load metadata from csv ######################################
df = pd.read_csv(os.path.join(data_path, "adni_dx_suvr_clean.csv"))
df = df.fillna('')
tmp = []
for i in range(len(ids_train)):
    id = ids_train[i]
    if '.' in id:
        id = id.split('.')
        dx = df[(df['RID'] == int(id[0])) & (df['MRI ImageID'] == int(id[1]))]['DX'].values[0]
    else:
        dx = df[(df['RID'] == int(id)) & (df['MRI ImageID'] == "")]['DX'].values[0]
    # train on AD/MCI/NL ([1,2,3]) or only AD/NL ([1,3])
    if dx in [1, 3]: tmp.append(ids_train[i])
ids_train = np.array(tmp)
tmp = []
for i in range(len(ids_val)):
    id = ids_val[i]
    if '.' in id:
        id = id.split('.')
        dx = df[(df['RID'] == int(id[0])) & (df['MRI ImageID'] == int(id[1]))]['DX'].values[0]
    else:
        dx = df[(df['RID'] == int(id)) & (df['MRI ImageID'] == "")]['DX'].values[0]
    # train on AD/MCI/NL ([1,2,3]) or only AD/NL ([1,3])
    if dx in [1, 3]: tmp.append(ids_val[i])
ids_val = np.array(tmp)
print(len(ids_train), len(ids_val))


batch_size = 20

# define dataset settings: support different data: mri/gray/white/csf/suvr/age/gender/edu/apoe. See dataset for details ##############################
adni_dataset_train = ADNI(os.path.join(data_path, "adni_dx_suvr_clean.csv"), ids_train, data_path, mri=True, hippo=True, grey=False, dx=True, age=True, gender=True, edu=True, apoe=True, split=1, size=(50, 50, 50), transform=True)
adni_dataset_val = ADNI(os.path.join(data_path, "adni_dx_suvr_clean.csv"), ids_val, data_path, mri=True, hippo=True, grey=False, dx=True, age=True, gender=True, edu=True, apoe=True, split=1, size=(50, 50, 50), transform=False)
dataloader_train = DataLoader(adni_dataset_train, batch_size=batch_size, num_workers=1, shuffle=True, pin_memory=True)
dataloader_val = DataLoader(adni_dataset_val, batch_size=batch_size, num_workers=1, shuffle=False, pin_memory=True)


num_epochs = 3000

learning_rate = 1e-6 # for focal / cross-entropy


classifier = Model().cuda()
classifier = nn.DataParallel(classifier)

evaluation = True
if evaluation:
    classifier.load_state_dict(torch.load("/home/chenwy/GV/suvr/jobs/saved_models/clf_ad1nl0_mri50_hippo30_lrflip_lenet_10.17.18.focal5.lr1e4.best.pth"))

criterion = FocalLoss(gamma=5)
# criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(classifier.parameters(), lr=learning_rate, weight_decay=1e-5)


if not evaluation:
    writer = SummaryWriter(log_dir=os.path.join(log_path, task_name))
    f_log = open(log_path + task_name + ".log", 'w')

# metrics = ConfusionMatrix(3)
metrics = ConfusionMatrix(2)
best_pred = 0
best_pred_acc = None
for epoch in range(num_epochs):
    classifier.train()
    # latest model use the below uncommented setting: mri + age + gender + edu + apoe as input ######################
    for i_batch, sample_batched in enumerate(tqdm(dataloader_train)):
        if evaluation: break
        images, lefts, rights, ages, genders, edus, apoes, labels = Variable(sample_batched['mri']).cuda(), Variable(sample_batched['left']).cuda(), Variable(sample_batched['right']).cuda(), Variable(sample_batched['age']).cuda(), Variable(sample_batched['gender']).cuda(), Variable(sample_batched['edu']).cuda(), Variable(sample_batched['apoe']).cuda(), Variable(sample_batched['dx']).view(-1).cuda()
        # ===================forward====================
        outputs = classifier(images, lefts, rights)
        
        loss = criterion(outputs, labels)
        
        predictions = torch.argmax(outputs, dim=1) # outputs are N * 2: 0 / 1
        # ===================backward====================
        optimizer.zero_grad()
        loss.backward()
        
        optimizer.step()
        metrics.update(labels.data.cpu().numpy(), predictions.data.cpu().numpy())
    results_train = metrics.get_scores()
    torch.cuda.empty_cache()
    metrics.reset()
    # =================== evaluation ========================
    if epoch % 1 == 0:
        classifier.eval()
        with torch.no_grad():
            for i_batch, sample_batched in enumerate(tqdm(dataloader_val)):
                images, lefts, rights, ages, genders, edus, apoes, labels = Variable(sample_batched['mri']).cuda(), Variable(sample_batched['left']).cuda(), Variable(sample_batched['right']).cuda(), Variable(sample_batched['age']).cuda(), Variable(sample_batched['gender']).cuda(), Variable(sample_batched['edu']).cuda(), Variable(sample_batched['apoe']).cuda(), Variable(sample_batched['dx']).view(-1).cuda()
                # ===================forward====================
                outputs = classifier(images, lefts, rights)
                
                predictions = torch.argmax(outputs, dim=1) # outputs are N * 2: 0 / 1
                metrics.update(labels.data.cpu().numpy(), predictions.data.cpu().numpy())
        results_val = metrics.get_scores()
        torch.cuda.empty_cache()
        log = ""
        log = log + 'epoch [{}/{}] train:'.format(epoch+1, num_epochs) +  str(results_train['accuracy']) + "  " + str(results_train['accuracy_mean']) + "\n"
        log = log + 'epoch [{}/{}] validation:'.format(epoch+1, num_epochs) +  str(results_val['accuracy']) + "  " + str(results_val['accuracy_mean']) + "\n"
        print(log)
        if evaluation: break
        writer.add_scalars('accuracy_mean', {'train': results_train['accuracy_mean'], 'validation': results_val['accuracy_mean']}, epoch)
        metrics.reset()
        torch.save(classifier.state_dict(), os.path.join(model_path, task_name + ".pth"))
        if results_val['accuracy_mean'] >= best_pred:
                best_pred = results_val['accuracy_mean']
                best_pred_acc = results_val['accuracy']
                # print(best_pred, best_pred_acc)
                torch.save(classifier.state_dict(), os.path.join(model_path, task_name + ".best.pth"))
        log = log + "best_pred: " + str(best_pred) + "  best_pred_acc: " + str(best_pred_acc) + "\n"
        log += "================================\n"
        f_log.write(log)
        f_log.flush()
f_log.close()
