# Modified from StyleGAN3 codebase

"""Streaming images and labels from datasets created with dataset_tool.py."""

from fnmatch import fnmatch
import os
import numpy as np
import zipfile
import PIL.Image
import json
import torch
import dnnlib
import cv2
import ipdb
import matplotlib.pyplot as plt
try:
    import pyspng
except ImportError:
    pyspng = None

#----------------------------------------------------------------------------

class Dataset(torch.utils.data.Dataset):
    def __init__(self,
        name,                   # Name of the dataset.
        raw_shape,              # Shape of the raw image data (NCHW).
        max_size    = None,     # Artificially limit the size of the dataset. None = no limit. Applied before xflip.
        use_labels  = False,    # Enable conditioning labels? False = label dimension is zero.
        xflip       = False,    # Artificially double the size of the dataset via x-flips. Applied after max_size.
        random_seed = 0,        # Random seed to use when applying max_size.
        pose_cond   = False,    # use pose conditioning or not
        pose_cond_type   = False,    # use pose conditioning or not
        clothtype_cond   = False,    # use clothing type conditioning or not
        conditional_D    = False,    # use conditional discriminator or not type conditioning or not
        colorcond    = False,        # use condition for the color
    ):
        self._name = name
        self._raw_shape = list(raw_shape)
        self._use_labels = use_labels
        self._raw_labels = None
        self._label_shape = None
        self._pose_cond = pose_cond
        self.pose_cond_type = pose_cond_type
        self.clothtype_cond = clothtype_cond
        self.conditional_D = conditional_D
        self.colorcond = colorcond
        # ipdb.set_trace()

        # Apply max_size.
        self._raw_idx = np.arange(self._raw_shape[0], dtype=np.int64)
        if (max_size is not None) and (self._raw_idx.size > max_size):
            np.random.RandomState(random_seed).shuffle(self._raw_idx)
            self._raw_idx = np.sort(self._raw_idx[:max_size])

        # Apply xflip.
        self._xflip = np.zeros(self._raw_idx.size, dtype=np.uint8)
        if xflip:
            self._raw_idx = np.tile(self._raw_idx, 2)
            self._xflip = np.concatenate([self._xflip, np.ones_like(self._xflip)])

    def _get_raw_labels(self):
        if self._raw_labels is None:
            self._raw_labels = self._load_raw_labels() if self._use_labels else None
            if self._raw_labels is None:
                self._raw_labels = np.zeros([self._raw_shape[0], 0], dtype=np.float32)
            assert isinstance(self._raw_labels, np.ndarray)
            assert self._raw_labels.shape[0] == self._raw_shape[0]
            assert self._raw_labels.dtype in [np.float32, np.int64]
            if self._raw_labels.dtype == np.int64:
                assert self._raw_labels.ndim == 1
                assert np.all(self._raw_labels >= 0)
        return self._raw_labels

    def close(self): # to be overridden by subclass
        pass

    def _load_raw_image(self, raw_idx): # to be overridden by subclass
        raise NotImplementedError

    def _load_raw_labels(self): # to be overridden by subclass
        raise NotImplementedError

    def __getstate__(self):
        return dict(self.__dict__, _raw_labels=None)

    def __del__(self):
        try:
            self.close()
        except:
            pass

    def __len__(self):
        return self._raw_idx.size

    def __getitem__(self, idx):
        image = self._load_raw_image(self._raw_idx[idx])
        # print(self._raw_idx[idx])
        # body_shape = np.load('/ps/project/tag_3d/data/CAPE_data/data/cape_release/minimal_body_shape/00032/00032_minimal.npy').astype(np.float32) #np.zeros([6890, 3]).astype(np.float32)
        # body_pose = np.zeros(72).astype(np.float32)
        if self.pose_cond_type == 'vposer':
            body_shape, body_pose, body_Vpose = self._load_raw_body_params(self._raw_idx[idx])
        else:
            body_shape, body_pose = self._load_raw_body_params(self._raw_idx[idx])
        cam = self._load_raw_camera(self._raw_idx[idx])
        assert isinstance(image, np.ndarray)
        assert list(image.shape) == self.image_shape
        assert image.dtype == np.uint8 # np.float32 #   
        if self._xflip[idx]:
            assert image.ndim == 3 # CHW
            image = image[:, :, ::-1]
        # if self._pose_cond:
        #     return image.copy(), body_pose, body_shape, body_pose, cam

        # Making images with alpha to white background
        img_mask = image[3:, :, :]/255.
        image[:3, :, :] = (image[:3, :, :] * img_mask) + ((np.ones_like(image[:3, :, :]) * 255.) * (1-img_mask))        
        
        if self.conditional_D:
            image = np.concatenate((image, self._load_raw_normals(self._raw_idx[idx])), 0)
        if self.pose_cond_type == 'vposer':
            return image.copy(), body_Vpose, body_shape, body_pose, cam

        return image.copy(), self.get_label(idx), body_shape, body_pose, cam

    def get_label(self, idx):
        if self.clothtype_cond:
            label = self._load_raw_labels_clothtype(self._raw_idx[idx])
        else:
            label = self._get_raw_labels()[self._raw_idx[idx]]
        if label.dtype == np.int64:
            onehot = np.zeros(6, dtype=np.float32) #np.zeros(self.label_shape, dtype=np.float32)
            onehot[label] = 1
            label = onehot
        if self.colorcond:
            color_feat = self._load_raw_labels_colortype(self._raw_idx[idx])
            label = np.hstack((label,color_feat))
        return label.copy()

    def get_shape_pose(self, idx):
        body_shape, body_pose = self._load_raw_body_params(self._raw_idx[idx])
        return body_shape, body_pose

    def get_details(self, idx):
        d = dnnlib.EasyDict()
        d.raw_idx = int(self._raw_idx[idx])
        d.xflip = (int(self._xflip[idx]) != 0)
        d.raw_label = self._get_raw_labels()[d.raw_idx].copy()
        return d

    @property
    def name(self):
        return self._name

    @property
    def image_shape(self):
        return list(self._raw_shape[1:])

    @property
    def num_channels(self):
        assert len(self.image_shape) == 3 # CHW
        return self.image_shape[0]

    @property
    def resolution(self):
        assert len(self.image_shape) == 3 # CHW
        assert self.image_shape[1] == self.image_shape[2]
        return self.image_shape[1]

    @property
    def label_shape(self):
        if self._label_shape is None:
            if self.clothtype_cond:
                if self.colorcond:
                    return np.array([6 + 512])
                return np.array([6])
            else:
                raw_labels = self._get_raw_labels()
            if raw_labels.dtype == np.int64:
                self._label_shape = [int(np.max(raw_labels)) + 1]
            else:
                self._label_shape = raw_labels.shape[1:]
        return list(self._label_shape)

    @property
    def label_dim(self):
        assert len(self.label_shape) == 1
        return self.label_shape[0]

    @property
    def has_labels(self):
        return any(x != 0 for x in self.label_shape)

    @property
    def has_onehot_labels(self):
        return self._get_raw_labels().dtype == np.int64

#----------------------------------------------------------------------------

class ImageFolderDataset(Dataset):
    def __init__(self,
        path,                   # Path to directory or zip.
        resolution      = None, # Ensure specific resolution, None = highest available.
        **super_kwargs,         # Additional arguments for the Dataset base class.
    ):
        self._path = path
        self._zipfile = None

        if os.path.isdir(self._path):
            self._type = 'dir'
            self._all_fnames = {os.path.relpath(os.path.join(root, fname), start=self._path) for root, _dirs, files in os.walk(self._path) for fname in files}
        elif self._file_ext(self._path) == '.zip':
            self._type = 'zip'
            self._all_fnames = set(self._get_zipfile().namelist())
        else:
            raise IOError('Path must point to a directory or zip')
        
        PIL.Image.init()
        # self._image_fnames = sorted(fname for fname in self._all_fnames if self._file_ext(fname)=='.npy')
        self._image_fnames_temp = sorted(fname for fname in self._all_fnames if self._file_ext(fname) in PIL.Image.EXTENSION)
        
        self._body_shape_fnames = sorted(fname for fname in self._all_fnames if fname.split('_')[-1]=='shape.json')
        self._body_pose_fnames = sorted(fname for fname in self._all_fnames if fname.split('_')[-1]=='pose.json')
        self._body_camR_fnames = sorted(fname for fname in self._all_fnames if fname.split('_')[-1]=='camR.json')
        self._body_camT_fnames = sorted(fname for fname in self._all_fnames if fname.split('_')[-1]=='camT.json')
        self._body_Vposer_fnames = sorted(fname for fname in self._all_fnames if fname.split('_')[-1]=='Vposer.json')
        # if self.clothtype_cond:
        self._clothlabels_fnames = sorted(fname for fname in self._all_fnames if fname.split('_')[-1]=='clothinglabel.json')
        self._colorlabels_fnames = sorted(fname for fname in self._all_fnames if fname.split('_')[-1]=='colorcondition.json')
        self._normal_fnames = sorted(fname for fname in self._all_fnames if fname.split('_')[-1]=='normal.png')
        self._image_fnames = sorted(list(set(self._image_fnames_temp) - set(self._normal_fnames)))

        # ipdb.set_trace()
        if len(self._image_fnames) == 0:
            raise IOError('No image files found in the specified path')

        name = os.path.splitext(os.path.basename(self._path))[0]
        raw_shape = [len(self._image_fnames)] + list(self._load_raw_image(0).shape)
        if resolution is not None and (raw_shape[2] != resolution or raw_shape[3] != resolution):
            raise IOError('Image files do not match the specified resolution')
        super().__init__(name=name, raw_shape=raw_shape, **super_kwargs)

    @staticmethod
    def _file_ext(fname):
        return os.path.splitext(fname)[1].lower()

    def _get_zipfile(self):
        assert self._type == 'zip'
        if self._zipfile is None:
            self._zipfile = zipfile.ZipFile(self._path)
        return self._zipfile

    def _open_file(self, fname):
        if self._type == 'dir':
            return open(os.path.join(self._path, fname), 'rb')
        if self._type == 'zip':
            return self._get_zipfile().open(fname, 'r')
        return None

    def close(self):
        try:
            if self._zipfile is not None:
                self._zipfile.close()
        finally:
            self._zipfile = None

    def __getstate__(self):
        return dict(super().__getstate__(), _zipfile=None)

    def _load_raw_image(self, raw_idx):
        fname = self._image_fnames[raw_idx]
        # print(fname)
        with self._open_file(fname) as f:
            if pyspng is not None and self._file_ext(fname) == '.png':
                image = pyspng.load(f.read())
            else:
                # image = np.array(PIL.Image.open(f))
                image = np.load(f)
        if image.ndim == 2:
            image = image[:, :, np.newaxis] # HW => HWC
        
        # ## TODO: Comment the following line after checking if blurring the alpha channel for real images works or not 
        # image[:, :, 3] = cv2.GaussianBlur(image[:, :, 3], (5,5), 0)

        image = image.transpose(2, 0, 1) # HWC => CHW
        return image

    def _load_raw_normals(self, raw_idx):
        fname = self._normal_fnames[raw_idx]
        # print(fname)
        with self._open_file(fname) as f:
            if pyspng is not None and self._file_ext(fname) == '.png':
                normal_image = pyspng.load(f.read())
            else:
                # normal_image = np.array(PIL.Image.open(f))
                normal_image = np.load(f)

        normal_image = normal_image.transpose(2, 0, 1) # HWC => CHW
        return normal_image

    def _load_raw_labels_clothtype(self, raw_idx):
        fname = self._clothlabels_fnames[raw_idx]

        with self._open_file(fname) as f:
            label = np.array(json.load(f)).astype(np.int64)
        
        return label
    
    def _load_raw_labels_colortype(self, raw_idx):
        fname = self._colorlabels_fnames[raw_idx]

        with self._open_file(fname) as f:
            colorcond = np.array(json.load(f)).astype(np.float32)
        
        return colorcond


    def _load_raw_labels(self):
        fname = 'dataset.json'
        if fname not in self._all_fnames:
            return None
        with self._open_file(fname) as f:
            labels = json.load(f)['labels']
        if labels is None:
            return None
        labels = dict(labels)
        labels = [labels[fname.replace('\\', '/')] for fname in self._image_fnames]
        labels = np.array(labels)
        labels = labels.astype({1: np.int64, 2: np.float32}[labels.ndim])
        return labels

    def _load_raw_body_params(self, raw_idx):
        fname_shape = self._body_shape_fnames[raw_idx]
        fname_pose = self._body_pose_fnames[raw_idx]

        with self._open_file(fname_shape) as f:
            body_shape = np.array(json.load(f))

        with self._open_file(fname_pose) as f:
            body_pose = np.array(json.load(f))

        # body_shape = np.array(json.load(open(fname_shape, 'r')))
        # body_pose = np.array(json.load(open(fname_pose, 'r')))

        if self.pose_cond_type == 'vposer':
            fname_Vpose = self._body_Vposer_fnames[raw_idx]
            with self._open_file(fname_Vpose) as f:
                body_Vpose = np.array(json.load(f))

            return body_shape.astype(np.float32), body_pose.astype(np.float32), body_Vpose.astype(np.float32)

        return body_shape.astype(np.float32), body_pose.astype(np.float32) # np.zeros_like(body_pose.astype(np.float32)) # 

    def _load_raw_camera(self, raw_idx):
        fname_cam = self._image_fnames[raw_idx]
        cam = int(0) # int(fname_cam.split('/')[-1].split('.')[0].split('mg')[-1]) # 

        # fname_camR = self._body_camR_fnames[raw_idx]
        # fname_camT = self._body_camT_fnames[raw_idx]

        # with self._open_file(fname_camR) as f:
        #     body_camR = np.array(json.load(f))

        # with self._open_file(fname_camT) as f:
        #     body_camT = np.array(json.load(f))

        # cam = np.concatenate((body_camR, np.expand_dims(body_camT, 1)), 1).astype(np.float32)

        return cam

#----------------------------------------------------------------------------
