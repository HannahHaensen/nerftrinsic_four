import torch
import torch.nn as nn
from torch.nn import init

from utils.lie_group_helper import make_c2w
from collections import OrderedDict
import numpy as np

from utils.training_utils import load_ckpt_to_net
import os

"""
https://github.com/ActiveVisionLab/nerfmm
"""

class LearnPoseGF(nn.Module):
    def __init__(self, num_cams, init_c2w=None, pose_encoding=False, embedding_scale=10):
        """
        :param num_cams: number of camera poses
        :param init_c2w: (N, 4, 4) torch tensor
        :param pose_encoding True/False, positional encoding or gaussian fourer
        :param embedding_scale hyperparamer, can also be adapted
        """
        super(LearnPoseGF, self).__init__()
        self.num_cams = num_cams
        self.embedding_size = 256
        self.all_points = torch.tensor([(i) for i in range(num_cams)])
        if init_c2w is not None:
            self.init_c2w = nn.Parameter(init_c2w, requires_grad=False)
        else:
            self.init_c2w = None
        self.lin1 = nn.Linear(self.embedding_size * 2, 64)
        self.gelu1 = nn.GELU()
        self.lin2 = nn.Linear(64, 64)
        self.gelu2 = nn.GELU()
        self.lin3 = nn.Linear(64, 6)

        self.embedding_scale = embedding_scale

        if pose_encoding:
            print("AXIS")
            posenc_mres = 5
            self.b = 2. ** np.linspace(0, posenc_mres, self.embedding_size // 2) - 1.
            self.b = self.b[:, np.newaxis]
            self.b = np.concatenate([self.b, np.roll(self.b, 1, axis=-1)], 0) + 0
            self.b = torch.tensor(self.b).float()
            self.a = torch.ones_like(self.b[:, 0])
        else:
            print("FOURIER")
            self.b = np.random.normal(loc=0.0, scale=self.embedding_scale,
                                      size=[self.embedding_size, 1])  # * self.embedding_scale
            self.b = torch.tensor(self.b).float()
            self.a = torch.ones_like(self.b[:, 0])

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.b = self.b.to(self.device)
        self.a = self.a.to(self.device)
        self.all_points = self.all_points.to(self.device)

    def forward(self, cam_id):
        """
        :param cam_id: current camera
        """
        cam_id = self.all_points[cam_id]
        cam_id = cam_id.unsqueeze(0)

        fourier_features = torch.concat([self.a * torch.sin((2. * torch.pi * cam_id) @ self.b.T),
                                         self.a * torch.cos((2. * torch.pi * cam_id) @ self.b.T)],
                                        axis=-1) / torch.linalg.norm(self.a)
        pred = self.lin1(fourier_features)
        pred = self.gelu1(pred)
        pred = self.lin2(pred)
        pred = self.gelu2(pred)
        pred = self.lin3(pred).squeeze(0)
        c2w = make_c2w(pred[:3], pred[3:])  # (4, 4)
        if self.init_c2w is not None:
            c2w = c2w @ self.init_c2w[cam_id][0]

        return c2w




