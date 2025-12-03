import warnings
import torch
import torch.nn as nn
from torch.testing._internal.common_nn import NNTestCase
import numpy as np
import pickle


class TestNN(NNTestCase):

    def test_parameterlistdict_setting_attributes(self):
        with warnings.catch_warnings(record=True) as w:
            map_param = map(nn.Parameter, [torch.rand(2), torch.rand(2)])
            nn.ParameterList(map_param)
            self.assertEqual(len(w) == 0, True)


fx = TestNN()
fx.test_parameterlistdict_setting_attributes()
