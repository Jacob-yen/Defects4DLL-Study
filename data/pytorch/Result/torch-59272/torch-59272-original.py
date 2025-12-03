import os
from io import BytesIO
from torch.package import EmptyMatchError, Importer, PackageExporter
from util import PackageTestCase
from torch.package.package_exporter import PackagingError

class TestDependencyAPI(PackageTestCase):
    def test_invalid_import(self):
        buffer = BytesIO()
        with self.assertRaises(PackagingError) as e:
            with PackageExporter(buffer, verbose=False) as exporter:
                exporter.save_source_string('foo', 'from ........ import lol')


fx = TestDependencyAPI()
fx.test_invalid_import()
