from .blender import BlenderDataset
from .llff import LLFFDataset
from .MB import MBDataset
dataset_dict = {'blender': BlenderDataset,
                'llff': LLFFDataset,
                "MB": MBDataset}