import os
import random
import numpy as np
from typing import Union
from pathlib import Path
from typing import Union, List

def get_all_hsr_file_paths(data_path: Union[str, Path]) -> List[Path]:
    filepaths = []

    if not isinstance(data_path, Path):
        data_path = Path(data_path)

    filenames = [x for x in data_path.glob(f"*.nii.gz") if x.is_file() and not x.name.startswith(".")]
    filepaths.extend(filenames)

    return sorted(filepaths)

def set_all_seeds(seed):
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)