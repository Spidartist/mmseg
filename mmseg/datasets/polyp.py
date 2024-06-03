from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset


@DATASETS.register_module()
class PolypDataset(BaseSegDataset):

    METAINFO = dict(
        classes=('background', 'non-background'),
        palette=[[0, 0, 0], [255, 255, 255]]
    )

    def __init__(self,
                 **kwargs) -> None:
        super().__init__(img_suffix='.png', seg_map_suffix='.png', reduce_zero_label=False, **kwargs)
