import torch
import unittest
import tempfile


class TestOldSerialization(unittest.TestCase):

    def test_serialization_zipfile_actually_jit(self):
        with tempfile.NamedTemporaryFile() as f:
            torch.jit.save(torch.jit.script(torch.nn.Linear(3, 4)), f)
            f.seek(0)
            torch.load(f)


fx = TestOldSerialization()
fx.test_serialization_zipfile_actually_jit()
