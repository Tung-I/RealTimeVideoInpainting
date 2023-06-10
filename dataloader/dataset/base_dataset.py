import torch
from torch.utils.data import Dataset
import pathlib 


class BaseDataset(Dataset):
    """The base class for all datasets.
    Args:
        data_dir: The directory of the saved data.
        type: The type of the dataset ('train', 'valid' or 'test').
        debug: Reduce the number of data in each epoch if true 
    """
    def __init__(
        self,
        base_dir: pathlib.Path,
        type: str,
        debug: bool = False,
        device: torch.device = None
    ):
        super().__init__()
        self.base_dir = base_dir
        self.type = type
        self.debug = debug
        self.device = device

