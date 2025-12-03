import torch
import math
from unittest import TestCase
from torch import nn
from torch.nn.utils import clip_grad_norm_


class TestNNDeviceType(TestCase):

    def test_clip_grad_norm_multi_device(self):
        devices0 = 'cuda:0'
        devices1 = 'cuda:1'


        class TestModel(nn.Module):

            def __init__(self):
                super(TestModel, self).__init__()
                self.layer1 = nn.Linear(10, 10)
                self.layer2 = nn.Linear(10, 10)
        test_model = TestModel()
        test_model.layer1.to(devices0)
        test_model.layer2.to(devices1)
        ref_model = TestModel().to(devices0)
        for norm_type in [2.0, math.inf]:
            for p in test_model.parameters():
                p.grad = torch.ones_like(p)
            for p in ref_model.parameters():
                p.grad = torch.ones_like(p)
            norm = clip_grad_norm_(test_model.parameters(), 0.5, norm_type=
                norm_type)


fx = TestNNDeviceType()
fx.test_clip_grad_norm_multi_device()
