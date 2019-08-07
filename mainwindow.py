from PyQt5.QtWidgets import QApplication, QLabel, QWidget, QLineEdit, QTextEdit, QPushButton
from PyQt5.QtWidgets import QGridLayout, QDesktopWidget
from PyQt5.QtWidgets import QFileDialog, QMessageBox
from PyQt5.QtWidgets import QProgressBar
from PyQt5.QtWidgets import QMainWindow
from PyQt5 import uic

import os
import sys
import time
import torch
import numpy as np
import pandas as pd

from datasets.adni_dataset import ADNI
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

from models.lenet_2conv_clf_oct_17_2018 import Lenet3D as Model

qtCreatorFile = 'app.ui'
Ui_MainWindow, QtBaseClass = uic.loadUiType(qtCreatorFile)

class MainWindow(QMainWindow, Ui_MainWindow):

    def __init__(self):

        QMainWindow.__init__(self)
        Ui_MainWindow.__init__(self)

        self.setupUi(self)
        self.testFileButton.clicked.connect(lambda: self.selectListFile('Test'))
        self.metaFileButton.clicked.connect(self.selectMetaFile)
        self.setDataPathButton.clicked.connect(self.setDataPath)

        self.modelFileButton.clicked.connect(self.setModelPath)

        self.initModelButton.clicked.connect(self.initModel)
        self.setDataIdButton.clicked.connect(self.setDataId)
        self.startButton.clicked.connect(self.start)

    def selectListFile(self, split):
        fname = QFileDialog.getOpenFileName(self, 'Open {} File'.format(split), '~')

        if fname[0]:
            print(fname[0])
            try:
                if split == 'Train':
                    self.ids_train = np.load(fname[0])
                    self.trainFileLabel.setText('Selected Train List File: {}'.format(fname[0].split('/')[-1]))
                elif split == 'Test':
                    #print(np.load(fname[0]))
                    self.ids_val = np.load(fname[0])
                    print(self.ids_val)
                    self.testFileLabel.setText('Selected Test List File: {}'.format(fname[0].split('/')[-1]))
            except:
                QMessageBox.critical(self, 'Error', 'Please choose a correct list file. ')

    def selectMetaFile(self):
        fname = QFileDialog.getOpenFileName(self, 'Open Meta File', '~')

        if fname[0]:
            try:

                df = pd.read_csv(fname[0])
                df = df.fillna('')
                '''
                tmp = []
                self.progressbar.setRange(0, len(self.ids_train))
                for i in range(len(self.ids_train)):
                    id = self.ids_train[i]
                    if '.' in id:
                        id = id.split('.')
                        dx = df[(df['RID'] == int(id[0])) & (df['MRI ImageID'] == int(id[1]))]['DX'].values[0]
                    else:
                        dx = df[(df['RID'] == int(id)) & (df['MRI ImageID'] == "")]['DX'].values[0]
                    # train on AD/MCI/NL ([1,2,3]) or only AD/NL ([1,3])
                    if dx in [1, 3]: tmp.append(self.ids_train[i])
                    self.progressbar.setValue(i)
                self.ids_train = np.array(tmp)
                '''
                tmp = []
                #self.progressbar.setRange(0, len(self.ids_val))
                for i in range(len(self.ids_val)):
                    id = self.ids_val[i]
                    if '.' in id:
                        id = id.split('.')
                        dx = df[(df['RID'] == int(id[0])) & (df['MRI ImageID'] == int(id[1]))]['DX'].values[0]
                    else:
                        dx = df[(df['RID'] == int(id)) & (df['MRI ImageID'] == "")]['DX'].values[0]
                    # train on AD/MCI/NL ([1,2,3]) or only AD/NL ([1,3])
                    if dx in [1, 3]: tmp.append(self.ids_val[i])
                    #self.progressbar.setValue(i)
                self.ids_val = np.array(tmp)
                self.numTotalLabel.setText("/{}".format(len(self.ids_val)))
                self.metaFileLabel.setText('Selected Meta File: {}'.format(fname[0].split('/')[-1]))
                self.metafile = fname[0]
                self.chooseNumberEdit.setText("1")
            except:
                QMessageBox.critical(self, 'Error', 'Please choose a correct meta file. ')
    '''
    def showFileListDiaglog(self):

        fname = QFileDialog.getOpenFileName(self, 'Open file', '~')

        if fname[0]:
            self.selectedFile.setText("Selected: " + fname[0].split('/')[-1])

            df = pd.read_csv(fname)
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
            print(len(ids_train), len(ids_val)
    '''
    def setDataPath(self):
        fname = QFileDialog.getExistingDirectory(self, 'Open Directory', '~')
        self.dataPath = fname
        self.dataPathLabel.setText("Selected Data Path: {}".format(fname))
    def setModelPath(self):

        fname = QFileDialog.getOpenFileName(self, 'Open Meta File', '~')
        if fname[0]:
            self.modelPath = fname[0]

        self.modelFileLabel.setText("Selected Pretrained Path: {}".format(fname[0].split('/')[-1]))
        #print(self.modelPath)
    def initModel(self):

        self.model = Model()
        state_dict = torch.load('clf_ad1nl0_mri50_hippo30_lrflip_lenet_10.17.18.focal5.lr1e4.best.pth', map_location="cpu")
        # create new OrderedDict that does not contain `module.`
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v

        self.model.load_state_dict(new_state_dict)
        self.model.eval()
        #try:
        #    self.model.load_state_dict(self.modelPath)
        #except:
        #    QMessageBox.critical(self, 'Error', 'Please choose a correct model file. ')

    def start(self):
        with torch.no_grad():
            for sample_batched in self.dataloader_val:
                images, lefts, rights, ages, genders, edus, apoes, labels = Variable(
                    sample_batched['mri']), Variable(sample_batched['left']), Variable(
                    sample_batched['right']), Variable(sample_batched['age']), Variable(
                    sample_batched['gender']), Variable(sample_batched['edu']), Variable(
                    sample_batched['apoe']), Variable(sample_batched['dx']).view(-1)
                # ===================forward====================
                outputs = self.model(images, lefts, rights)

                predictions = torch.argmax(outputs, dim=1)
                print(predictions)

        self.classificationResultLabel.setText("Classification result: {}".format(predictions.item()))

    def setDataId(self):
        ids = [self.ids_val[int(self.chooseNumberEdit.text()) - 1]]
        if os.path.exists(self.dataPath):
            self.adni_dataset_val = ADNI(self.metafile, ids, self.dataPath, mri=True,
                                         hippo=True, grey=False, dx=True, age=True, gender=True, edu=True, apoe=True,
                                         split=1, size=(50, 50, 50), transform=False)
            self.dataloader_val = DataLoader(self.adni_dataset_val, batch_size=1, num_workers=1, shuffle=False)


    def center(self):
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())