import os

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import imageio

from utils.pose_utils import center_poses
from utils.lie_group_helper import convert3x4_4x4


def load_imgs(image_dir, img_ids, res_ratio, HWFocal):
    '''

    :param image_dir: directory of images in scene
    :param img_ids: ids of train/val images
    :param res_ratio: res ratio for BLEFF e.g. 2
    :return: list of images ond their names
    '''
    img_names = np.array(sorted(os.listdir(image_dir)))  # all image names
    img_names = img_names[img_ids]  # image name for this split

    img_paths = [os.path.join(image_dir, n) for n in img_names]

    img_list = []
    idx = 0
    for p in tqdm(img_paths):
        img = imageio.imread(p)[:, :, :3]  # (H, W, 3) np.uint8
        new_h = img.shape[0] // res_ratio
        new_w = img.shape[1] // res_ratio
        img = torch.Tensor(img).unsqueeze(dim=0)
        img = img.permute(0, 3, 1, 2)  # (N, 3, H, W)

        img = F.interpolate(img, size=(new_h, new_w), mode='bilinear')  # (N, 3, new_H, new_W)
        img = img.permute(0, 2, 3, 1).squeeze(dim=0)  # (N, new_H, new_W, 3)
        img_list.append(img / 255)  # List with H, W, 3 in float format
        idx += 1
        for i in range(len(img_names)):
            if str(str(new_h) + str(new_w)) not in HWFocal:
                HWFocal[str(new_h) + str(new_w)] = [len(HWFocal)]

    return img_list, img_names, HWFocal


def load_split(scene_dir, img_dir, data_type, num_img_to_load, skip, res_ratio):
    '''
    load pre-splitted train/val ids
    :param scene_dir: dir to dataset scene
    :param img_dir: dir to images
    :param data_type:
    :param num_img_to_load:
    :param skip: skip X number of images
    :param res_ratio: resize ratio of images
    :return: data for train/val
    '''
    img_ids_train = np.loadtxt(os.path.join(scene_dir, 'train' + '_ids.txt'), dtype=np.int32, ndmin=1)
    img_ids_val = np.loadtxt(os.path.join(scene_dir, 'val' + '_ids.txt'), dtype=np.int32, ndmin=1)
    img_ids = img_ids_train
    if data_type == 'val':
        img_ids == img_ids_val

    if num_img_to_load == -1:
        img_ids = img_ids[::skip]
        print('Loading all available {0:6d} images'.format(len(img_ids)))
    elif num_img_to_load > len(img_ids):
        print('Required {0:4d} images but only {1:4d} images available. '
              'Exit'.format(num_img_to_load, len(img_ids)))
        exit()
    else:
        img_ids = img_ids[:num_img_to_load:skip]

    N_imgs = img_ids.shape[0]


    HWFocal = {}
    # load images
    imgs, img_names, HWFocal = load_imgs(img_dir, img_ids, res_ratio, HWFocal)  # (N, H, W, 3) torch.float32
    if data_type == 'val':
        _, _, HWFocal = load_imgs(img_dir, img_ids_train, res_ratio, HWFocal)  # (N, H, W, 3) torch.float32
    else:
        _, _, HWFocal = load_imgs(img_dir, img_ids_val, res_ratio, HWFocal)  # (N, H, W, 3) torch.float32


    result = {
        'imgs': imgs,  # (N, H, W, 3) torch.float32
        'img_names': img_names,  # (N, )
        'N_imgs': N_imgs,
        'img_ids': img_ids,  # (N, ) np.int
        'HWFocal': HWFocal
    }
    return result


def read_meta(in_dir, use_ndc):
    """
    Read the poses_bounds_original.npy file produced by LLFF imgs2poses.py.
    This function is modified from https://github.com/kwea123/nerf_pl.
    """
    poses_bounds = np.load(os.path.join(in_dir, 'poses_bounds.npy'))  # (N_images, 17)

    c2ws = poses_bounds[:, :15].reshape(-1, 3, 5)  # (N_images, 3, 5)
    bounds = poses_bounds[:, -2:]  # (N_images, 2)

    H, W, focal = [], [], []
    for elem in c2ws[:, :, -1]:
        H.append(int(elem[0]))
        W.append(int(elem[1]))
        focal.append(float(elem[2]))
    # H, W, focal = c2ws[:, :, -1]

    # correct c2ws: original c2ws has rotation in form "down right back", change to "right up back".
    # See https://github.com/bmild/nerf/issues/34
    c2ws = np.concatenate([c2ws[..., 1:2], -c2ws[..., :1], c2ws[..., 2:4]], -1)

    # (N_images, 3, 4), (4, 4)
    c2ws, pose_avg = center_poses(c2ws)  # pose_avg @ c2ws -> centred c2ws


    if use_ndc:
        # correct scale so that the nearest depth is at a little more than 1.0
        # See https://github.com/bmild/nerf/issues/34
        near_original = bounds.min()
        scale_factor = near_original * 0.75  # 0.75 is the default parameter
        # the nearest depth is at 1/0.75=1.33
        bounds /= scale_factor
        c2ws[..., 3] /= scale_factor

    c2ws = convert3x4_4x4(c2ws)  # (N, 4, 4)

    results = {
        'c2ws': c2ws,  # (N, 4, 4) np
        'bounds': bounds,  # (N_images, 2) np
        'H': H,  # scalar
        'W': W,  # scalar
        'focal': focal,  # scalar
        'pose_avg': pose_avg,  # (4, 4) np
    }
    return results


class DataloaderParameterLearning:
    """
    Most useful fields:
        self.c2ws:          (N_imgs, 4, 4)      torch.float32
        self.imgs           (N_imgs, H, W, 4)   torch.float32
        self.ray_dir_cam    (H, W, 3)           torch.float32
        self.H              scalar
        self.W              scalar
        self.N_imgs         scalar
    """

    def __init__(self, base_dir, scene_name, data_type, res_ratio, num_img_to_load, skip, use_ndc, load_img=True):
        """
        :param base_dir:
        :param scene_name:
        :param data_type:   'train' or 'val'.
        :param res_ratio:   int [1, 2, 4] etc to resize images to a lower resolution.
        :param num_img_to_load/skip: control frame loading in temporal domain.
        :param use_ndc      True/False, just centre the poses and scale them.
        :param load_img:    True/False. If set to false: only count number of images, get H and W,
                            but do not load imgs. Useful when vis poses or debug etc.
        """
        self.base_dir = base_dir
        self.scene_name = scene_name
        self.data_type = data_type
        self.res_ratio = res_ratio
        self.num_img_to_load = num_img_to_load
        self.skip = skip
        self.use_ndc = use_ndc
        self.load_img = load_img

        self.scene_dir = os.path.join(self.base_dir, self.scene_name)
        self.img_dir = os.path.join(self.scene_dir, 'images')

        # all meta info
        meta = read_meta(self.scene_dir, self.use_ndc)

        self.c2ws = meta['c2ws']  # (N, 4, 4) all camera pose
        self.H = meta['H']
        self.W = meta['W']
        self.focal = meta['focal']

        if isinstance(self.H, list):
            for i in range(len(self.H)):
                self.H[i] = self.H[i] // self.res_ratio
                self.W[i] = self.W[i] // self.res_ratio
                self.focal[i] /= self.res_ratio
        else:
            self.H = self.H // self.res_ratio
            self.W = self.W // self.res_ratio
            self.focal /= self.res_ratio

        self.near = 0.0
        self.far = 1.0

        HWFocalGT = {}
        for i in range(len(self.H)):
            if str(self.H[i]) + str(self.W[i]) not in HWFocalGT:
                HWFocalGT[str(self.H[i]) + str(self.W[i])] = [self.focal[i], len(HWFocalGT)]

        '''Load train/val split'''
        split_results = load_split(self.scene_dir, self.img_dir, self.data_type, self.num_img_to_load,
                                   self.skip, self.res_ratio)

        self.imgs = split_results['imgs']  # (N, H, W, 3) torch.float32
        self.img_names = split_results['img_names']  # (N, )
        self.N_imgs = split_results['N_imgs']
        self.img_ids = split_results['img_ids']  # (N, ) np.int

        self.HWFocalGT = HWFocalGT

        self.HWFocal = split_results['HWFocal']


if __name__ == '__main__':
    scene_name = 'LLFF/fern'
    use_ndc = True
    scene = DataloaderParameterLearning(base_dir='/your/data/path',
                                        scene_name=scene_name,
                                        data_type='train',
                                        res_ratio=8,
                                        num_img_to_load=-1,
                                        skip=1,
                                        use_ndc=use_ndc)
