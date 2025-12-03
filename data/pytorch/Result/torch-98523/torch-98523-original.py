import torch
import torch.utils.data.datapipes as dp
import torch.utils.data.graph
import torch.utils.data.graph_settings
from torch.testing._internal.common_utils import TestCase
from torch.utils.data import IterDataPipe


class CustomShardingIterDataPipe(IterDataPipe):
    def __init__(self, dp):
        self.dp = dp
        self.num_of_instances = 1
        self.instance_id = 0

    def apply_sharding(self, num_of_instances, instance_id):
        self.num_of_instances = num_of_instances
        self.instance_id = instance_id

    def __iter__(self):
        for i, d in enumerate(self.dp):
            if i % self.num_of_instances == self.instance_id:
                yield d


class NumbersDataset(IterDataPipe):
    def __init__(self, size=10):
        self.size = size

    def __iter__(self):
        yield from range(self.size)

    def __len__(self):
        return self.size


def _mul_10(x):
    return x * 10


def _mod_3_test(x):
    return x % 3 == 1


class TestSharding(TestCase):
    def _get_pipeline(self):
        numbers_dp = NumbersDataset(size=10)
        dp0, dp1 = numbers_dp.fork(num_instances=2)
        dp0_upd = dp0.map(_mul_10)
        dp1_upd = dp1.filter(_mod_3_test)
        combined_dp = dp0_upd.mux(dp1_upd)
        return combined_dp

    def test_legacy_custom_sharding(self):
        dp = self._get_pipeline()
        sharded_dp = CustomShardingIterDataPipe(dp)
        torch.utils.data.graph_settings.apply_sharding(sharded_dp, 3, 1)
        items = list(sharded_dp)
        self.assertEqual([1, 20], items)


fx = TestSharding()
fx.test_legacy_custom_sharding()
