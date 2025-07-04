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
from transformers import get_cosine_schedule_with_warmup

# MLOps
import wandb
from accelerate import Accelerator

# logging
from loguru import logger

# our stuff
from model import *
from data import *
from utils import plot_logger

R = Random(7)

class Trainer:
    def __init__(self, args, accelerator=None, run_id=None):
        # set up the trainer
        self.args = args
        if not accelerator:
            self.accelerator = Accelerator(
                log_with="wandb",
                gradient_accumulation_steps=self.args.accumulate_steps
            )
        else:
            self.accelerator = accelerator
        self.accelerator.init_trackers(
            project_name="adventure", 
            config=vars(args),
            init_kwargs={"wandb": {"mode": None if args.wandb else "disabled",
                                   "name": args.experiment,
                                   "resume": "allow",
                                   "id": run_id}},
        )
        self.plot, self.get_plots = plot_logger(accelerator=self.accelerator,
                                                args=self.args)

        # ...and the output path
        save_dir = Path(args.out_dir) / args.experiment
        save_dir.mkdir(parents=True, exist_ok=True)

        self.save_dir = save_dir / "checkpoint"
        self.best_dir = str(save_dir / "best")

        # <<<<<<< set up models <<<<<<<
        self.model = GPT(args)
        # >>>>>>> set up models >>>>>>>

        # <<<<<<< set up data <<<<<<<
        #    
        # self.train_dl = ...
        # self.val_dl = ...
        #    
        # >>>>>>> set up data >>>>>>>
        # leave blank
        # this will exist if we are resuming from checkpoint
        self.train_dl_skipped = None 

        # optimizer
        # <<<<<<< set up optimizer <<<<<<<
        # self.optim = self.model.configure_optimizers(
        #     weight_decay=args.weight_decay,
        #     learning_rate=args.lr,
        #     betas=(args.beta1, args.beta2),
        #     device_type="cuda" if torch.cuda.is_available() else "cpu"
        # )
        #
        # OR
        #
        # self.optim = AdamW(self.model.parameters(), lr=args.lr,
        #                    betas=(args.beta1, args.beta2),
        #                    weight_decay=args.weight_decay)
        # >>>>>>> set up optimizer >>>>>>>

        # compute training size + the counter (useful for mid-checkpoint recovery) 
        self.total_batches = len(self.train_dl)
        self.global_step_counter_ = 0
        self.best_val_score_ = float("-inf") # "score" means higher is better 

        # weeeeeeeeeeee
        (self.model, self.optim, self.train_dl, self.val_dl) = self.accelerator.prepare(
            self.model, self.optim, self.train_dl, self.val_dl)
        if self.accelerator.is_main_process and args.wandb:
            wandb.watch(self.model)

    def _vomit(self):
        """recursively vomit every method and attribute of self into a namespace

        only useful if you are Jack and has a weird Jupyter setup. I apologise
        for the this abuse of all that's good about Python.
        """

        from types import SimpleNamespace
        ns = SimpleNamespace()

        from torch.nn import ModuleList
        buffer = [(i, self) for i in dir(self) if i[0] != "_"]
        names = []
        while len(buffer) > 0:
            member, head = buffer.pop(-1)
            attr = getattr(head, member)
            ns.__dict__[member] = attr

            def include(attr):
                for j in dir(attr):
                    if j not in names and j[0] != "_":
                        buffer.append((j, attr))
                        names.append(j)

            # some special rules for including things 
            if isinstance(attr, ModuleList):
                for component in attr:
                    include(component)
            else:
                include(attr)

        return ns

    def train(self):
        for eid in range(self.args.epochs):
            if self.global_step_counter_ >= ((eid+1)*self.total_batches):
                logger.debug("SKIPPING EPOCH {} due to global step count...", eid)
                continue

            self.epoch()

        self.finish()

    def finish(self):
        self.accelerator.end_training()

    def val(self):
        with torch.inference_mode():
            # <<<<<<< do some validation <<<<<<<
            # 
            # # remeber, score is higher = better
            # score = self.gather(...).cpu().item()
            # metrics = { "val/metric": ... }
            # 
            # >>>>>>> do some validation >>>>>>>

            return score, metrics

    def epoch(self):
        if self.accelerator.is_main_process:
            logger.info("BEGIN EPOCH")

        # because sometimes the load function may skip some epochs
        dl = self.train_dl if not self.train_dl_skipped else self.train_dl_skipped
        for indx, i in enumerate(dl):
            # <<<<<<< do some setup <<<<<<<
            # >>>>>>> do some setup >>>>>>>

            # take a step, optionally with plotting
            if indx % self.args.plot_interval == 0:
                with self.plot(self.global_step_counter_, debug=(not self.args.wandb)):
                    loss, train_metrics = self.step(i, indx)
            else:
                loss, train_metrics = self.step(i, indx)
            train_metrics["train/lr"] = self.optim.param_groups[0]["lr"]

            # perform logging, and then increment
            # (we do this because global_step_counter_
            #  is useful not as the # of steps but how
            #  many we need to skip for warm start)
            if indx % self.args.report_interval == 0 and indx != 0:
                self.accelerator.log(train_metrics, step=self.global_step_counter_)
                if self.accelerator.is_main_process:
                    logger.info("TRAIN | {}/{} | loss {}", self.global_step_counter_,
                                self.total_batches*self.args.epochs, loss)
            self.global_step_counter_ += 1

            logger.debug("STEP | {} | {}", indx, train_metrics)

            # save a checkpoint, if needed
            if indx % self.args.checkpoint_interval == 0 and indx != 0:
                self.save(str(self.save_dir / str(self.global_step_counter_)))
            # perform validation and save a checkpoint, if needed
            if indx % self.args.validation_interval == 0 and indx != 0:
                score, val_metrics = self.val()
                self.accelerator.log(val_metrics, step=self.global_step_counter_)
                if self.accelerator.is_main_process:
                    logger.info("VAL | {} | score {}", self.global_step_counter_, score)

                if score > self.best_val_score_:
                    if self.accelerator.is_main_process:
                        logger.info("VAL | BEST SCORE | score {}", score)
                    self.best_val_score_ = score
                    self.save(self.best_dir)

        # we are done using the skipped DL since we finished the remaining batch
        self.train_dl_skipped = None

    def gradients(self, batch):
        # <<<<<<< do some work <<<<<<<
        # 
        # loss = self.model(**batch, ...)
        #
        # >>>>>>> do some work >>>>>>>

        self.accelerator.backward(loss/self.args.accumulate_steps)

        # <<<<<<< prepare metrics <<<<<<<
        # 
        # loss = self.gather(loss).cpu().item() 
        # metrics = { "train/loss": ... }
        #
        # >>>>>>> prepare metrics >>>>>>>

        return loss, metrics

    def step(self, batch, indx):
        if indx % self.args.accumulate_steps == 0:
            loss, metrics = self.gradients(batch)
            self.optim.step()
            # >>>>>>> scheduler shenanigans >>>>>>>
            # 
            # self.scheduler.step()
            # 
            # >>>>>>> scheduler shenanigans >>>>>>>
            self.optim.zero_grad()
        else:
            with self.accelerator.no_sync(self.model):
                loss, metrics = self.gradients(batch)

        return loss, metrics

    def load(self, path):
        self.accelerator.load_state(path)
        with open(os.path.join(path, "config.json"), 'r') as df:
            data = json.load(df)

        self.args = Namespace(**data.get("config", {}))
        self.global_step_counter_ = data.get("steps", 0)
        self.best_val_score_ = data.get("score", 0)

        # skip batches
        self.train_dl_skipped = self.accelerator.skip_first_batches(self.train_dl,
                                                                    self.global_step_counter_ % self.total_batches)

    def save(self, path):
        logger.debug("CHECKPOINT | saving checkpoint at {}", path)
        self.accelerator.save_state(path)
        with open(os.path.join(path, "config.json"), 'w') as df:
            json.dump({
                "config": vars(self.args),
                "steps": self.global_step_counter_,
                "score": self.best_val_score_,
                "wandb": wandb.run.id if self.args.wandb else None
            }, df)

    @classmethod
    def from_pretrained(cls, path, disable_wandb=True, accelerator=None):
        with open(os.path.join(path, "config.json"), 'r') as df:
            data = json.load(df)
        args = Namespace(**data.get("config", {}))
        args.wandb = False if disable_wandb else args.wandb
        new = cls(args, accelerator, run_id=data.get("wandb"))
        new.load(path)

        if disable_wandb:
            new.args.wandb = False

        return new

    @property
    def device(self):
        return self.accelerator.device

    def gather(self, n):
        result = self.accelerator.gather(n)
        if isinstance(result, list):
            return sum(result)/len(result)
        else:
            return result.mean()
    

