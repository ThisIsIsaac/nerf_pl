import torch
from torch.utils.data import Dataset
import glob
import numpy as np
import os
from PIL import Image
from torchvision import transforms as T
from .ray_utils import *
import cv2
import skimage.io
import os, torch, torch.utils.data as data
from PIL import Image
import numpy as np
import sys
import re
import pdb
import torchvision
import warnings
from llff import *
import cv2
warnings.filterwarnings('ignore', '.*output shape of zoom.*')
import os
import torch
import numpy as np
from PIL import Image
from rich import print
from rich import pretty
pretty.install()
from rich import traceback
traceback.install()
from kornia import create_meshgrid
import time



def get_ray_directions(H, W, K):
    grid = create_meshgrid(H, W, normalized_coordinates=False)[0]
    i, j = grid.unbind(-1)

    #* already in 2D homogenous form
    directions = torch.stack([i , j , torch.ones_like(i)], -1)
    K_inv = np.linalg.inv(K.astype(np.int))
    K_inv = np.tile(K_inv, (H, W, 1, 1))

    out = np.matmul(K_inv, directions[..., np.newaxis])

    return torch.squeeze(out.float())

IMG_EXTENSIONS = [
 '.jpg', '.JPG', '.jpeg', '.JPEG',
 '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP']

def normalize(v):
    """Normalize a vector."""
    return v / np.linalg.norm(v)


def readPFM(file):
    file = open(file, 'rb')

    header = file.readline().rstrip()
    if (sys.version[0]) == '3':
        header = header.decode('utf-8')
    if header == 'PF':
        color = True
    elif header == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    if (sys.version[0]) == '3':
        dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode('utf-8'))
    else:
        dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline())
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    if (sys.version[0]) == '3':
        scale = float(file.readline().rstrip().decode('utf-8'))
    else:
        scale = float(file.readline().rstrip())

    if scale < 0:  # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>'  # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    return data, scale


def default_loader(path):
    return Image.open(path).convert('RGB')

def disparity_loader(path):

    if path.endswith('.png'):
        data = Image.open(path)
        data = np.ascontiguousarray(data,dtype=np.float32)/256
        out = data
    elif path.endswith(".pfm"):
        out = readPFM(path)[0]
    elif path.endswith(".npy"):
        out = np.load(path)
    else:
        raise ValueError("Disparity file is in unrecognized format: " + path)

    for s in out.strides:
        if s < 0:
            out = out.copy()
        break
    return out

class MBDataset(data.Dataset):
    def __init__(self, root_dir, split = 'train', img_wh = (504, 378), spheric_poses = False, val_num = 1, feature_extractor = None,\
                process_img = (lambda x: x), loader=default_loader, dploader=disparity_loader):

        self.split = split
        self.img_wh = img_wh

        #* for when the cameras are all pointing at some central object of interest
        #* places this point at the origin and rescales the sphere of cameras so the average distance to the origin is 1
        #* https://github.com/bmild/nerf/issues/34#issuecomment-616607285
        self.spheric_poses = spheric_poses
        self.root_dir = root_dir
        self.loader = loader
        self.dploader = dploader
        self.feature_extractor = feature_extractor
        self.process_img = process_img
        self.val_num=max(1, val_num)
        self.white_back = False
        self.counter = 0
        self.define_transforms()
        self.read_meta()


    def read_cam(self, line):
        arr = [[],[],[]]
        i=0
        counter = 0
        line = line[1:len(line)-2]
        for w in line.split(" "):
            arr[i].append(np.float(w.strip("[]; ")))
            counter+=1
            if ";" in w:
                i +=1
            if counter == 9:
                break
        return np.array(arr)

    #? not called anywhere. why?
    def extract_feature(self, img_path):
        img = Image.open(img_path).convert("RGB")
        img = self.preprocess_img(img)
        self.feature_extractor.eval()
        with torch.no_grad():
            feature = self.feature_extractor(img)
        return feature

    # return focal length and baseline
    def read_calib(self, path):
        """
        https://vision.middlebury.edu/stereo/data/scenes2014/
        calib.txt:
            cam0=[3997.684 0 1176.728; 0 3997.684 1011.728; 0 0 1]
            cam1=[3997.684 0 1307.839; 0 3997.684 1011.728; 0 0 1]
            doffs=131.111
            baseline=193.001
            width=2964
            height=1988
            ndisp=280
            isint=0
            vmin=31
            vmax=257
            dyavg=0.918
            dymax=1.516

        max_Z = baseline * f / (vmin + doffs)
        then divide all 3 coordinate of points by max_Z
        """
        calib = {}
        with open(path) as f:
            for line in f.readlines():
                if "cam0" in line:
                    calib["cam0"] = self.read_cam(line.split("=")[1])
                elif "cam1" in line:
                    calib["cam1"] = self.read_cam(line.split("=")[1])

                elif "baseline" in line:
                    calib["baseline"] = np.float(line.split("=")[1])
                elif "width" in line:
                    calib["W"] = np.float(line.split("=")[1])
                elif "height" in line:
                    calib["H"] = np.float(line.split("=")[1])
                elif "ndisp" in line:
                    calib["ndisp"] = np.float(line.split("=")[1])
                if "vmin" in line:
                    vmin = np.float(line.split("=")[1])
                if "vmax" in line:
                    vmax = np.float(line.split("=")[1])
                if "doffs" in line:
                    doffs = np.float(line.split("=")[1])
                if "ndisp" in line:
                    ndisp = np.float(line.split("=")[1])

        calib["focal"] = np.array([calib["cam0"][0,0], calib["cam0"][1,1]])
        z_max=calib["baseline"] * calib["focal"][0] / (vmax + doffs)
        z_min = calib["baseline"] * calib["focal"][0] / (vmin + doffs)
        calib["z_max"] = z_max
        calib["z_min"] = z_min
        calib["max_disp"] = ndisp
        return calib

    def read_meta(self):
        start_time = time.time()
        self.counter+=1
        print("read_meta called (" + str(self.counter)+") times")

        calib = self.read_calib(os.path.join(self.root_dir, "calib.txt"))
        self.K = calib["cam0"]
        W, H = calib["W"], calib["H"]
        self.baseline = calib["baseline"]
        self.focal = calib["focal"]
        self.z_min = calib["z_min"]
        self.z_max = calib["z_max"] #* z_max is used to normalize ( divide all 3 axis of 3D points of the generated ray points)
        self.image_paths = [os.path.join(self.root_dir, "im0.png"),
                            os.path.join(self.root_dir, "im1.png")]
        self.disp_paths = [os.path.join(self.root_dir, "disp0GT.pfm")] * len(self.image_paths)
        self.max_disp = calib["max_disp"]
        self.bounds = np.array([self.z_min, self.z_max]) #* [mi, max depth]

        # Step 1: rescale focal length according to training resolution
        # H, W, self.focal = poses[0, :, -1]  # original intrinsics, same for all images
        assert H * self.img_wh[0] == W * self.img_wh[1], \
            f'You must set @img_wh to have the same aspect ratio as ({W}, {H}) !'

        self.focal *= self.img_wh[0] / W

        #! do we need this normalization step for MB? check where the pivot is for the extrinsic poses of LLFF
        #! no we don't.
        # Step 2: correct poses
        # Original poses has rotation in form "down right back", change to "right up back"
        # See https://github.com/bmild/nerf/issues/34
        self.poses = self.get_extrinsic_matrix(self.baseline)
        # self.poses = np.concatenate([poses[..., 1:2], -poses[..., :1], poses[..., 2:4]], -1)
        # (N_images, 3, 4) exclude H, W, focal
        # self.poses, self.pose_avg = center_poses(poses)
        # distances_from_center = np.linalg.norm(self.poses[..., 3], axis=1)
        val_idx = np.argmin(self.poses[..., 3])  # choose val image as the closest to
        # center image

        #! isn't this what we were trying to do with z_min?
        # Step 3: correct scale so that the nearest depth is at a little more than 1.0
        # See https://github.com/bmild/nerf/issues/34

        #! z_min is minimum disparity, which is maximum depth for us. and self.bounds.min() is trying to get the actual
        #! minimum depth. so self.bounds should contain maximum and minimum depth derived from z_min and z_max
        near_original = self.bounds.min()
        scale_factor = near_original * 0.75  # 0.75 is the default parameter
        # the nearest depth is at 1/0.75=1.33
        self.bounds /= scale_factor
        self.poses[..., 3] /= scale_factor

        # ray directions for all pixels, same for all images (same H, W, focal)
        self.directions = \
            get_ray_directions(self.img_wh[1], self.img_wh[0], self.K)  # (H, W, 3)

        if self.split == 'train':  # create buffer of all rays and rgb data
            # use first N_images-1 to train, the LAST is val
            self.all_rays = []
            self.all_rgbs = []
            self.all_disp = []
            self.all_masks = []
            self.ray_idx_to_disp_idx = []
            # self.train_img_paths = []
            for i, image_path in enumerate(self.image_paths):
                #! train on validation as well
                # if i == val_idx:  # exclude the val image
                #     continue
                c2w = torch.FloatTensor(self.poses[i])

                img = Image.open(image_path).convert('RGB')

                #* read and save disp
                gt_disp = self.dploader(self.disp_paths[i])
                gt_disp = cv2.resize(gt_disp, self.img_wh,
                                     interpolation=cv2.INTER_LINEAR)
                gt_disp = torch.from_numpy(gt_disp)
                gt_disp = gt_disp.view( -1)
                self.all_disp.append(gt_disp)

                #! instead of taking max_disp as CLI argument, we read it from a single calib file
                mask = (gt_disp > 0) & (gt_disp < self.max_disp)
                self.all_masks.append(mask)

                assert img.size[1] * self.img_wh[0] == img.size[0] * self.img_wh[1], \
                    f'''{image_path} has different aspect ratio than img_wh, 
                           please check your data!'''
                img = img.resize(self.img_wh, Image.LANCZOS)

                # * image preprocessing (no actual processing, just `ToTensor`)
                img = self.transform(img)  # (3, h, w)
                img = img.view(3, -1).permute(1, 0)  # (h*w, 3) RGB
                self.all_rgbs += [img]

                # make a mapping table from ray idx to disp idx
                self.ray_idx_to_disp_idx.extend([i]*(img.shape[0]))

                rays_o, rays_d = get_rays(self.directions, c2w)  # both (h*w, 3)
                if not self.spheric_poses:
                    near, far = 0, 1
                    rays_o, rays_d = get_ndc_rays(self.img_wh[1], self.img_wh[0],
                                                  self.focal[0], 1.0, rays_o, rays_d)
                    # near plane is always at 1.0
                    # near and far in NDC are always 0 and 1
                    # See https://github.com/bmild/nerf/issues/34
                else:
                    near = self.bounds.min()
                    far = min(8 * near, self.bounds.max())  # focus on central object only

                self.all_rays += [torch.cat([rays_o, rays_d,
                                             near * torch.ones_like(rays_o[:, :1]),
                                             far * torch.ones_like(rays_o[:, :1])],
                                            1)]  # (h*w, 8)

            self.all_rays = torch.cat(self.all_rays, 0)  # ((N_images-1)*h*w, 8)
            self.all_rgbs = torch.cat(self.all_rgbs, 0)  # ((N_images-1)*h*w, 3)
            self.all_disp = torch.cat(self.all_disp, 0)
            self.all_masks = torch.cat(self.all_masks, 0)

        elif self.split == 'val':
            print('val image is', self.image_paths[val_idx])
            self.c2w_val = self.poses[val_idx]
            self.image_path_val = self.image_paths[val_idx]
            self.disp_path_val = self.disp_paths[val_idx]


        else:  # for testing, create a parametric rendering path
            if self.split.endswith('train'):  # test on training set
                self.poses_test = self.poses
            elif not self.spheric_poses:
                focus_depth = 3.5  # hardcoded, this is numerically close to the formula
                # given in the original repo. Mathematically if near=1
                # and far=infinity, then this number will converge to 4
                radii = np.percentile(np.abs(self.poses[..., 3]), 90, axis=0)
                self.poses_test = create_spiral_poses(radii, focus_depth)
            else:
                radius = 1.1 * self.bounds.min()
                self.poses_test = create_spheric_poses(radius)
        # print("duration = " + str(time.time() - start_time))

    def define_transforms(self):
        self.transform = T.ToTensor()

    def __getitem__(self, idx):
        if self.split == 'train':  # use data in the buffers
            sample = {'rays': self.all_rays[idx],
                      'rgbs': self.all_rgbs[idx],
                      "mask": self.all_masks[self.ray_idx_to_disp_idx[idx]],
                      "gt_disp": self.all_disp[self.ray_idx_to_disp_idx[idx]]}
            return sample

        else:
            if self.split == 'val':
                c2w = torch.FloatTensor(self.c2w_val)

            else:
                c2w = torch.FloatTensor(self.poses_test[idx])

            rays_o, rays_d = get_rays(self.directions, c2w)
            if not self.spheric_poses:
                near, far = 0, 1
                rays_o, rays_d = get_ndc_rays(self.img_wh[1], self.img_wh[0],
                                              self.focal[0], 1.0, rays_o, rays_d)
            else:
                near = self.bounds.min()
                far = min(8 * near, self.bounds.max())

            rays = torch.cat([rays_o, rays_d,
                              near * torch.ones_like(rays_o[:, :1]),
                              far * torch.ones_like(rays_o[:, :1])],
                             1)  # (h*w, 8)

            sample = {'rays': rays,
                      'c2w': c2w}

            if self.split == 'val':
                img = Image.open(self.image_path_val).convert('RGB')
                img = img.resize(self.img_wh, Image.LANCZOS)
                img = self.transform(img)  # (3, h, w)
                img = img.view(3, -1).permute(1, 0)  # (h*w, 3)
                sample['rgbs'] = img

                gt_disp = self.dploader(self.disp_path_val)
                gt_disp = cv2.resize(gt_disp, self.img_wh, interpolation=cv2.INTER_LINEAR)
                gt_disp = torch.from_numpy(gt_disp)
                gt_disp = gt_disp.view( -1)
                mask = (gt_disp > 0) & (gt_disp < self.max_disp)
                sample["gt_disp"] = gt_disp
                sample["mask"] = mask
        return sample

    def get_extrinsic_matrix(self, baseline):
        no_transform = [[1, 0, 0, 0],
                        [0, 1, 0, 0],
                        [0, 0, 1, 0]]
        R = np.identity(3)
        T = np.array([[baseline, 0, 0]]).T
        E = np.concatenate([R, T], axis=1)
        return np.stack([no_transform, E], axis=0)
    def __len__(self):
        if self.split == 'train':
            return len(self.all_rays)
        if self.split == 'val':
            return self.val_num
        return len(self.poses_test)
