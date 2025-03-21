# common standard library utilities
import os
import sys
import time
import json
import math
import random
from random import Random

from pathlib import Path
from argparse import Namespace

# machine learning and data utilities
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.optim import AdamW
from torch.optim.lr_scheduler import SequentialLR, LinearLR

# huggingface
from transformers import AutoConfig, AutoModel, AutoTokenizer

# MLOps
import wandb
from accelerate import Accelerator

# logging
from loguru import logger
from utils import plot

class Model(nn.Module):
    # <<<<<<< you probably want a model <<<<<<<
    # plot("rooster/plot_chicken",
    #       thing_to_plot=torch.tensor(),
    #       key=self.layer_num)  # <- optional key to plot multiple things per cycle
    #                            #    such as activations per layer
    # >>>>>>> you probably want a model >>>>>>>

    raise NotImplementedError()


