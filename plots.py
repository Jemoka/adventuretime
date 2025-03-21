# common standard library utilities
import os
import sys
import time
import json
import math
import random
from random import Random
from collections import defaultdict

from pathlib import Path
from argparse import Namespace

# machine learning and data utilities
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

# huggingface
import logging
from loguru import logger

# to contextualize plotting
from contextlib import contextmanager

# plotting tools
import wandb
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

# setting style
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.5)
sns.set_palette(["#8C1515",    # Red
                 "#175E54",    # Green
                 "#E98300",    # Orange
                 "#007C92",    # Teal
                 "#DAD7CB",    # Light Gray
                 "#B83A4B",    # Cardinal Red
                 "#4D4F53"])   # Dark Gray

R = Random(7)

def sort_by_key(data, reverse=False):
    """small utility to sort data by key, which is a usual usecase"""
    sorted_list = [i[1] for i in sorted(list(data.items()),
                                        reverse=reverse,
                                        key=lambda a:a[0])]
    final_dict = defaultdict(list)

    for i in sorted_list:
        for k,v in i.items():
            final_dict[k].append(v)

    return dict(final_dict)

###############################################

# global fig size
FIGSIZE = (12,8)

# each of the plot functions should decide
# what its going to do, and return the correspdoing
# wandb object. For instance, if you are plotting
# text, wandb.Html. If you are plotting image,
# return wandb.Image, etc.
#
# you should return both the *PRE WRAP* object
# as well as the *POST WRAP* image; the former
# will be for saving, the latter will be for emitting
def plot_chicken(data):
    fig, ax = plt.subplots(figsize=FIGSIZE)

    # <<<<<<< extracting data <<<<<<<
    # extract a sorted ordering of layers' forking scores
    # data = sort_by_key(data)  # with key, multiple items per session OR
    # data = data  # without key, so single layer per session
    # >>>>>>> extracting data >>>>>>>

    # <<<<<<< make your plot <<<<<<<
    # 
    # sns.heatmap(data.cpu().detach(), 
    #       cmap='viridis',
    #       xticklabels='auto',
    #       yticklabels='auto',
    #       ax=ax)
    # plt.close(fig) # remember to call close against the figure!
    # scores = data  # what to return to the outer layer
    #
    # >>>>>>> set up models >>>>>>>

    return scores, wandb.Image(fig)

# what function to run for what function?
PLOTS = {
    "rooster/plot_chicken": plot_chicken
}

###############################################

def plot(name, data):
    plot_func = PLOTS.get(name)
    if plot_func == None:
        return None
    else:
        return plot_func(data)

    


