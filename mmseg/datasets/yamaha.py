from collections import OrderedDict
from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class YamahaDataset(CustomDataset):
    """Yamaha dataset.

    In segmentation map annotation for Yamaha, 0 stands for background, which
    is not included in 150 categories. ``reduce_zero_label`` is fixed to True.
    The ``img_suffix`` is fixed to '.jpg' and ``seg_map_suffix`` is fixed to
    '.png'.
    """
    LABEL_TO_COLOR = OrderedDict({
        "high_vegetation": [40, 80, 0],
        "traversable_grass": [128, 255, 0],
        "smooth_trail": [178, 176, 153],
        "obstacle": [255, 0, 0],
        "sky": [1, 88, 255],
        "rough_trial": [156, 76, 30],
        "puddle": [255, 0, 128],
        "non_traversable_low_vegetation": [0, 160, 0]
    })

    PALETTE = list(LABEL_TO_COLOR.values())
    CLASSES = list(LABEL_TO_COLOR.keys())

    def __init__(self, **kwargs):
        super(YamahaDataset, self).__init__(
            img_suffix='.jpg',
            seg_map_suffix='.png',
            reduce_zero_label=True,
            **kwargs)
