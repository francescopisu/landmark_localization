import os
from pathlib import Path
from yacs.config import CfgNode as CN

_C = CN()

# configuration files
_C.CONFIG = CN()
_C.CONFIG.CONFIG_PATH = "config"

# data
_C.DATA = CN()
_C.DATA.BASE_DIRNAME = "data"
_C.DATA.OUTPUT_PATH = "data/output"
_C.DATA.DATA_PATH = "/Users/francescopisu/Workspace/Research/Projects/CoroCTAiomics/notebooks/data"
_C.DATA.REFERENCES_PATH = "data/OSTIA_HSR/reference.csv"

_C.DATA.TEST_SIZE = 0.3

# training
_C.TRAIN = CN()
_C.TRAIN.CHECKPOINTS_PATH = "data/output/checkpoints"
_C.TRAIN.LOGS_PATH = "data/output/training_logs"
_C.TRAIN.PLOTS_PATH = "data/output/learning_curves"

# network-specific configurations
_C.TRAIN.CENTERLINE_NET = CN()
_C.TRAIN.CENTERLINE_NET.MAX_EPOCHS = 100

_C.TRAIN.OSTIUM_NET = CN()
_C.TRAIN.OSTIUM_NET.MAX_EPOCHS = 1

_C.TRAIN.SEED_NET = CN()
_C.TRAIN.SEED_NET.MAX_EPOCHS = 1

# misc
_C.MISC = CN()
_C.MISC.SEED = 1303


def get_defaults():
    return _C.clone()
