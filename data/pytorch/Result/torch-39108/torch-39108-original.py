import torch
import unittest
import tempfile
import os
import torch.distributed as c10d
import copy
import torch.nn as nn


class DistributedDataParallelSingleProcessTest(unittest.TestCase):

    def setUp(self):
        self.rank = 0
        self.world_size = 1
        self.file = tempfile.NamedTemporaryFile(delete=False)

    def _test_base(self, net, inp, check_allclose=True):
        store = c10d.FileStore(self.file.name, self.world_size)
        process_group = c10d.ProcessGroupGloo(store, self.rank, self.world_size
            )
        nn.parallel.DistributedDataParallel(copy.deepcopy(net),
            process_group=process_group)

    def test_cuda(self):
        self._test_base(nn.Linear(2, 2).to(0), [torch.randn(30, 2).to(0)])


fx = DistributedDataParallelSingleProcessTest()
fx.setUp()
fx.test_cuda()
