from typing import Union
from pathlib import Path

def get_all_hsr_file_paths(data_path: Union[str, Path],
                           data_dirname: str) -> List[Path]:
    filepaths = []

    if not isinstance(data_path, Path):
        data_path = Path(data_path)

    filenames = [x for x in data_path.glob(f"**/{data_dirname}/*.nii.gz") if x.is_file() and not x.name.startswith(".")]
    filepaths.extend(filenames)

    return sorted(filepaths)
