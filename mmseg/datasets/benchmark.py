from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset


@DATASETS.register_module()
class BenchmarkDataset(BaseSegDataset):

    METAINFO = dict(
        classes=('background', 'damaged'),
        palette=[[0, 0, 0], [255, 255, 255]])

    def __init__(self,
                 **kwargs) -> None:
        super().__init__(**kwargs)