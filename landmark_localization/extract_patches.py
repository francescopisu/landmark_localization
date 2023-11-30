import numpy as np
from config.defaults import get_defaults 
from landmark_localization.utils import get_all_hsr_file_paths

def extract_patches(image: np.ndarray, cut_size: int = 9):
    pass
    
    
if __name__ == "__main__":
    # for each exam: extract patches at each ostium and save them in dedicated folders
    cfg = get_defaults()
    file_paths = get_all_hsr_file_paths()