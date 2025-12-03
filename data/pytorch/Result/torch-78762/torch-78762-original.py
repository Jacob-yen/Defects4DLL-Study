from torch.utils.data import DataLoader, DataLoader2
from torch.utils.data.datapipes.iter import IterableWrapper
from torch.testing._internal.common_utils import TestCase


class TestDataLoader2(TestCase):

    def test_basics(self):
        dp = IterableWrapper(list(range(1000))).sharding_filter()
        dl_list = list(DataLoader(dp, batch_size=3, collate_fn=lambda x: x, num_workers=2))
        dl2_list = list(DataLoader2(dp, batch_size=3, collate_fn=lambda x: x, num_workers=2))
        self.assertEqual(dl_list, dl2_list)


fx = TestDataLoader2()
fx.test_basics()
