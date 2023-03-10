import torch
import torch.nn as nn

class LearnFocalCamDependent(nn.Module):
    def __init__(self, num_cams=1, req_grad=True, fx_only=False, order=2):
        """
        :param num_cams: number of cameras
        :param req_grad: True/False, train intrinsic parameters
        :param fx_only: True/False optimize only for fx, if false optimize fx,fy
        :param order
        """
        super(LearnFocalCamDependent, self).__init__()
        self.fx_only = fx_only  # If True, output [fx, fx]. If False, output [fx, fy]
        self.order = order
        self.num_cams = num_cams
        print(self.num_cams)
        if self.fx_only:
            self.fx = nn.Parameter(torch.ones(size=(num_cams, 1), dtype=torch.float32), requires_grad=req_grad)
        else:
            self.fx = nn.Parameter(torch.ones(size=(num_cams, 1), dtype=torch.float32), requires_grad=req_grad)
            self.fy = nn.Parameter(torch.ones(size=(num_cams, 1), dtype=torch.float32), requires_grad=req_grad)


    def forward(self, i=None, H=None, W=None):  # the i=None is just to enable multi-gpu training
        """
            :param i
            :param H: int image height
            :param W: int image width
        """
        if self.fx_only:
            if self.order == 2:
                fxfy = torch.stack([self.fx[i] ** 2 * W, self.fx[i] ** 2 * W])
            else:
                fxfy = torch.stack([self.fx[i] * W, self.fx[i] * W])
        else:
            if self.order == 2:
                fxfy = torch.stack([self.fx[i]**2 * W, self.fy[i]**2 * H])
            else:
                fxfy = torch.stack([self.fx[i] * W, self.fy[i] * H])
        return fxfy
