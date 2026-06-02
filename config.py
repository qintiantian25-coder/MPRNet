#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Configuration management using yacs CfgNode.
"""

from typing import Any, List
from yacs.config import CfgNode as CN


class Config(object):
    r"""
    A collection of all the required configuration parameters. This class is a nested dict-like
    structure, with nested keys accessible as attributes. It contains sensible default values for
    all the parameters, which may be overridden by (first) through a YAML file and (second) through
    a list of attributes and values.

    Parameters
    ----------
    config_yaml: str
        Path to a YAML file containing configuration parameters to override.
    config_override: List[Any], optional (default= [])
        A list of sequential attributes and values of parameters to override. This happens after
        overriding from YAML file.

    Examples
    --------
    >>> _C = Config("config.yaml", ["OPTIM.BATCH_SIZE", 2048, "BETA", 0.7])
    >>> _C.ALPHA  # default: 100.0
    1000.0
    """

    def __init__(self, config_yaml: str, config_override: List[Any] = []):

        self._C = CN()
        self._C.GPU = [0]
        self._C.VERBOSE = False

        self._C.MODEL = CN()
        self._C.MODEL.MODE = 'BlindPixel'
        self._C.MODEL.SESSION = 'MPRNet_blind'
        self._C.MODEL.IN_C = 3
        self._C.MODEL.OUT_C = 3
        self._C.MODEL.N_FEAT = 80
        self._C.MODEL.SCALE_UNETFEATS = 48
        self._C.MODEL.SCALE_ORSNETFEATS = 32
        self._C.MODEL.NUM_CAB = 8
        self._C.MODEL.KERNEL_SIZE = 3
        self._C.MODEL.REDUCTION = 4
        self._C.MODEL.BIAS = False

        self._C.OPTIM = CN()
        self._C.OPTIM.BATCH_SIZE = 16
        self._C.OPTIM.VAL_BATCH_SIZE = 8
        self._C.OPTIM.NUM_EPOCHS = 3000
        self._C.OPTIM.WARMUP_EPOCHS = 3
        self._C.OPTIM.LR_INITIAL = 0.0002
        self._C.OPTIM.LR_MIN = 0.000001
        self._C.OPTIM.BETA1 = 0.5

        self._C.TRAINING = CN()
        self._C.TRAINING.VAL_AFTER_EVERY = 20
        self._C.TRAINING.RESUME = False
        self._C.TRAINING.SAVE_IMAGES = False
        self._C.TRAINING.DATA_ROOT = '/home/student_server/Qtt/NAFNet/data_new'
        self._C.TRAINING.SAVE_DIR = './experiments'
        self._C.TRAINING.TRAIN_PS = 256
        self._C.TRAINING.VAL_PS = 256
        self._C.TRAINING.NUM_WORKERS = 8
        self._C.TRAINING.NUM_WORKERS_VAL = 4
        self._C.TRAINING.USE_MASK_WEIGHT = False
        self._C.TRAINING.MASK_BLIND_WEIGHT = 5.0
        self._C.TRAINING.EDGE_LOSS_WEIGHT = 0.05

        self._C.TEST = CN()
        self._C.TEST.CHECKPOINT = './experiments/models/best_model.pt'
        self._C.TEST.SAVE_DIR = './results'
        self._C.TEST.IMAGE_BORDER = 0

        # Override parameter values from YAML file first, then from override list.
        self._C.merge_from_file(config_yaml)
        self._C.merge_from_list(config_override)

        # Make an instantiated object of this class immutable.
        self._C.freeze()

    def dump(self, file_path: str):
        r"""Save config at the specified file path."""
        self._C.dump(stream=open(file_path, "w"))

    def __getattr__(self, attr: str):
        return self._C.__getattr__(attr)

    def __repr__(self):
        return self._C.__repr__()
