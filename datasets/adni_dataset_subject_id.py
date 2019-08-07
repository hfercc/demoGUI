import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from skimage import transform
import os
from random import choice
from scipy.ndimage.filters import gaussian_filter

class ADNI(Dataset):
    """ADNI Dataset."""

    def __init__(self, csv_file, ids, data_path, rid2label=None, mri=False, grey=False, white=False, csf=False, pet=False, suvr=False, dx=False, age=False, gender=False, edu=False, apoe=False, size=None, hippo=False, split=1, multi_modal=False, transform=False, noise=False):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.metadata = pd.read_csv(csv_file)
        self.metadata = self.metadata.fillna('')
        self.ids = ids
        self.data_path = data_path
        self.rid2label = rid2label
        self.mri = mri
        self.grey = grey
        self.white = white
        self.csf = csf
        self.pet = pet
        self.suvr = suvr
        self.dx = dx
        self.age = age
        self.gender = gender
        self.edu = edu
        self.apoe = apoe
        self.apoe_dict = {"NC": 0, "HT": 1, "HM": 2}
        self.size = size
        self.hippo = hippo
        self.split = split
        self.multi_modal = multi_modal
        self.transform = transform
        self.noise = noise
        # grey_mi = np.load("/hdd1/chenwy/adni/mi_grey_scaled.npy")
        # self.grey_roi = (grey_mi >= 0.005).astype('int')
        # self.grey_offroi = (grey_mi < 0.005).astype('int')

    def __len__(self):
        return len(self.ids)
    
    def _transform_shift(self, image):
        # https://pytorch.org/docs/stable/torchvision/transforms.html
        # https://discuss.pytorch.org/t/torchvision-transfors-how-to-perform-identical-transform-on-both-image-and-target/10606/6
        shift_range = list(range(-2, 3))
        shift_x = choice(shift_range)
        shift_y = choice(shift_range)
        shift_z = choice(shift_range)
        image = np.roll(image, shift_x, 0)
        if shift_x >= 0: image[:shift_x, :, :] = 0
        else: image[shift_x:, :, :] = 0
        image = np.roll(image, shift_y, 1)
        if shift_y >= 0: image[:, :shift_y, :] = 0
        else: image[:, shift_y:, :] = 0
        image = np.roll(image, shift_z, 2)
        if shift_z >= 0: image[:, :, :shift_z] = 0
        else: image[:, :, shift_z:] = 0
        
        return image
    
    def _transform_gaussian(self, image, sigma=None):

        if not sigma: sigma = np.random.rand() * 0.5
        image = gaussian_filter(image * 1.0, sigma=sigma)
        # image = image * self.grey_roi + gaussian_filter(image * 1.0, sigma=sigma) * self.grey_offroi
        
        return image
    
    def _transform_sagittal_flip(self, image):
        image = np.flip(image, 0)
        return image
    
    def _transform_noise(self, image):
        shape = image.shape
        # image = image + np.random.randn(shape[0], shape[1], shape[2])
        image = image + np.random.randn(*shape).astype('float32')
        return image

    def _transform(self, image):
        # image = self._transform_shift(image)

        # if np.random.random() > 0.5:
        #     image = self._transform_gaussian(image, sigma=1.2)
        if np.random.random() > 0.5:
            image = self._transform_sagittal_flip(image).copy()

        # image = self._transform_noise(image)
        return image
    
    def chunk2channel(self, image, split=2):
        '''image: h*w*d'''
        shape = image.shape
        chunks_x = np.split(image, split, axis=0)
        chunks_xy = []
        for chunk in chunks_x:
            chunks_xy += np.split(chunk, split, axis=1)
        chunks_xyz = []
        for chunk in chunks_xy:
            chunks_xyz += np.split(chunk, split, axis=2)
        chunks = np.zeros((split**3, shape[0]//split, shape[1]//split, shape[2]//split)).astype('float32')
        for i in range(len(chunks_xyz)):
            chunks[i] = chunks_xyz[i]
        return chunks
    
    def get_image(self, id, name):
        ''' name: mri/grey/white/pet '''
        '''[8:112, 10:138, :112] OR [8:113, 10:139, :114]'''
        if self.split == 3:
            image = np.load(os.path.join(self.data_path, name, id + "." + name + ".npy"))[8:113, 10:139, :114]
        else:
            image = np.load(os.path.join(self.data_path, name, id + "." + name + ".npy"))[8:112, 10:138, :112]
        # if self.size is not None:
        #     image = transform.resize(image, self.size, mode='constant')
        image = image.astype('float32')
        if self.transform:
            image = self._transform(image)
        image /= image.max()
        if self.split > 1: image = self.chunk2channel(image, split=self.split)
        # else: image = np.expand_dims(image, axis=0) # add one channel dimension: (1, 104, 128, 112)
        return image
    
    def get_multi_modal(self, id, name):
        image = np.load(os.path.join(self.data_path, name, id + "." + name + ".npy"))[8:112, 10:138, :112]
        image /= image.max()
        image = image.astype('float32')
        
        transformed = self._transform(image)
        cropped = image[24:74, 49:75, 24:70]
        if self.split > 1:
            image = self.chunk2channel(image, self.split)
            transformed = self.chunk2channel(transformed, self.split)
            cropped = self.chunk2channel(cropped, self.split)
        else:
            image = np.expand_dims(image, axis=0) # add one channel dimension: (1, 104, 128, 112)
            transformed = np.expand_dims(transformed, axis=0) # add one channel dimension: (1, 104, 128, 112)
        return image, transformed, cropped

    def get_hippo(self, image):
        # left hippocampus as [64:90, 59:85, 24:50] and right hippocampus as [31:57, 59:85, 24:50] in [121, 145, 121]
        # 30,30,30 => [54:84, 47:77, 22:52] and [21:51, 47:77, 22:52] in [104, 128, 112] ([8:112, 10:138, :112])
        left = image[54:84, 47:77, 22:52]
        right = image[21:51, 47:77, 22:52]
        return left, right

    def __getitem__(self, id):
        id = self.ids[id]
        sample = {'id': int(str(id)) if '.' not in id else int(str(id.split('.')[0]))}
        if self.rid2label != None:
            sample['id'] = self.rid2label[sample['id']]
        
        if self.mri:
            # mri_origin = np.load(os.path.join(self.data_path, "mri", id + ".npy"))[8:112, 10:138, :112]
            sample['mri'] = self.get_image(id, 'mri')
            if self.noise:
                sample['mri_noise'] = self._transform_noise(sample['mri'])
            if self.hippo:
                sample['left'], sample['right'] = self.get_hippo(sample['mri'])
                sample['left'] = np.expand_dims(sample['left'], axis=0).astype('float32')
                sample['right'] = np.expand_dims(sample['right'], axis=0).astype('float32')
            if self.size is not None:
                sample['mri'] = transform.resize(sample['mri'], self.size, mode='constant')
            sample['mri'] = np.expand_dims(sample['mri'], axis=0).astype('float32')
            
        if self.grey:
            if self.multi_modal:
                image, transformed, cropped = self.get_multi_modal(id, 'grey')
                sample['grey'] = image
                sample['grey_transform'] = transformed
                sample['grey_hippo'] = cropped
            else:
                sample['grey'] = self.get_image(id, 'grey')
                if self.transform:
                    sample['grey'] = self._transform(sample['grey'])
                if self.noise:
                    sample['grey_noise'] = self._transform_noise(sample['grey'])
            
        if self.white:
            sample['white'] = self.get_image(id, 'white')
            if self.transform:
                sample['white'] = self._transform(sample['white'])
            
        if self.csf:
            sample['csf'] = self.get_image(id, 'csf')
            if self.transform:
                sample['csf'] = self._transform(sample['csf'])
            
        if self.pet:
            sample['pet'] = self.get_image(id, 'pet')
        

        rid = int(id.split('.')[0])
        if '.' in id: image_id = int(id.split('.')[1])
        else: image_id = ''

        if self.suvr:
            sample['suvr'] = self.metadata[(self.metadata['RID'] == rid) & (self.metadata['MRI ImageID'] == image_id)]['SUVR'].values.astype('float32')
        
        if self.dx:
            # 1=NL 2=MCI, 3=AD
            sample['dx'] = self.metadata[(self.metadata['RID'] == rid) & (self.metadata['MRI ImageID'] == image_id)]['DX'].values.astype('float32')
            sample['dx'] -= 1
            if sample['dx'][0] == 2: sample['dx'][0] = 1

        if self.age:
            sample['age'] = self.metadata[(self.metadata['RID'] == rid) & (self.metadata['MRI ImageID'] == image_id)]['Age'].values.astype('float32')
            sample['age'] /= 100.
        
        if self.gender:
            # 1=male, 2=female
            sample['gender'] = self.metadata[(self.metadata['RID'] == rid) & (self.metadata['MRI ImageID'] == image_id)]['Gender'].values.astype('float32')
            sample['gender'] -= 1

        if self.edu:
            sample['edu'] = self.metadata[(self.metadata['RID'] == rid) & (self.metadata['MRI ImageID'] == image_id)]['Education'].values.astype('float32')
            sample['edu'] /= 25.

        if self.apoe:
            sample['apoe'] = self.metadata[(self.metadata['RID'] == rid) & (self.metadata['MRI ImageID'] == image_id)]['ApoE4'].values[0]
            sample['apoe'] = np.array(self.apoe_dict[sample['apoe']]).reshape(1).astype('float32')
            sample['apoe'] /= 3
        
        # label = self.metadata.loc[self.metadata["RID"] == rid]["label"].values
        # # label = label.astype('int')
        # label = label.astype('float32')
        
        # sample = {'image': image, 'label': label}
        # if self.train:
        #     # sample = {'image_noise': image_noise, 'image': image}
        #     sample = {'image_noise': self.chunk2channel(image_noise), 'image': self.chunk2channel(image)}
        # else:
        #     # sample = {'image': image}
        #     sample = {'image': self.chunk2channel(image)}
        # sample = {'left': left, 'right': right, 'label': label}
        # sample = {'left1': left1, 'right1': right1, 'left2': left2, 'right2': right2, 'left3': left3, 'right3': right3, 'left4': left4, 'right4': right4, 'label': label}
        # sample = {'grey': grey, 'white': white, 'label': label}

        return sample
