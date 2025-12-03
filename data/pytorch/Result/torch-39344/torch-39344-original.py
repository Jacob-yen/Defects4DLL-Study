import torch
from unittest import TestCase
import numpy as np
import pickle
class TestNNInit(TestCase):

    def test_convert_sync_batchnorm(self):
        module = torch.nn.Sequential(torch.nn.BatchNorm1d(100, affine=True),
            torch.nn.InstanceNorm1d(100)).cuda()
        comp_module = torch.nn.Sequential(torch.nn.BatchNorm1d(100, affine=
            True), torch.nn.InstanceNorm1d(100)).cuda()
        comp_module.load_state_dict(module.state_dict())
        sync_bn_module = torch.nn.SyncBatchNorm.convert_sync_batchnorm(module)
        layer = list(comp_module.children())[0]
        converted_layer = list(sync_bn_module.children())[0]
        key = list(layer.state_dict().keys())[0]
        self.assertEqual(layer.state_dict()[key].device, converted_layer.state_dict()[key].device)


testClass = TestNNInit()
testClass.test_convert_sync_batchnorm()
