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

# MLOps
import wandb
from accelerate import Accelerator, DeepSpeedPlugin
from accelerate.state import AcceleratorState

# logging
import logging
from loguru import logger

# to contextualize plotting
from contextlib import contextmanager

# plot function
from plots import plot as do_plot
from collections.abc import Callable

R = Random(7)

class PlotLoggingHandler(logging.Handler):
    def __init__(self, *args, **kwargs):
        self.args = kwargs.pop("args")
        self.accelerator = kwargs.pop("accelerator")
        super().__init__(*args, **kwargs)

        # because we will plot() whenever, but only plot
        # specificlally when we actually want to plot
        self.__write_plot = False
        # things yet to be plotted --- plot_name : {key : value}
        # we plot only when the plot() function is called
        self.__cached_plots = defaultdict(lambda : defaultdict(dict))
        # we truly save every plot now
        # {plot: [(plot, idx)]}
        self.__saved_plots = defaultdict(list)

    @property
    def plots(self):
        return dict(self.__saved_plots)

    def arm(self):
        self.__write_plot = True
    def plot(self, idx=None, override_plot_funcs={}):
        # actually emit the plots
        logs = {}
        for k,v in self.__cached_plots.items():
            v = {i: {b: c() if isinstance(c, Callable) else c for b,c in a.items()} for i,a in v.items()}
            if override_plot_funcs.get(k):
                plotted = override_plot_funcs.get(k)(v)
            else:
                plotted = do_plot(k, v)
            if not plotted:
                continue
            if isinstance(plotted, dict):
                # we want multiple plots from each plot function
                for a,b in plotted.items(): 
                    (save,log) = b
                    logs[a] = log
                    self.__saved_plots[a].append((save,idx))
            else:
                (save,log) = plotted
                logs[k] = log
                self.__saved_plots[k].append((save,idx))
        self.accelerator.log(logs, step=idx)

        # flush the cache
        self.__cached_plots = defaultdict(lambda : defaultdict(dict))
        self.__write_plot = False

    def emit(self, record: logging.LogRecord) -> None:
        name = record.getMessage()
        kwargs = record.extra["payload"]
        key = record.extra["key"]

        if self.__write_plot:
            if key == None:
                if self.__cached_plots.get(name):
                    logger.warning("Plot already exists before flush! Make sure that "
                                   "if you are plotting multiple things with the same "
                                   "name within a single plot context, that the have "
                                   "distinct keys; overwriting the past one!!!; duplicated "
                                   "name: {}", name)
                self.__cached_plots[name] = kwargs
            else:
                self.__cached_plots[name][key] = kwargs

def plot_logger(*args, **kwargs):
    handler = PlotLoggingHandler(*args, **kwargs)
    logger.add(handler,
               filter=lambda x:x["extra"].get("task", "") == "plot",
               format="{message}")

    @contextmanager
    def emit(idx=None, override_plot_funcs={}):
        handler.arm()
        try:
            yield
        finally:
            # this will actually emit the plots
            handler.plot(idx,override_plot_funcs)
    def get_plots():
        return handler.plots

    return emit, get_plots
        
def plot(name, key=None, **kwargs):
    logger.bind(task="plot", payload=kwargs, key=key).info(name)



