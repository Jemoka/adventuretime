import os
import sys
import argparse

import torch
import random
import numpy as np
import inspect
import logging
from loguru import logger
from dotenv import load_dotenv

import parameters
from commands import execute

load_dotenv()

logger.remove()

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

class InterceptHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        # Get corresponding Loguru level if it exists.
        level: str | int
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # Find caller from where originated the logged message.
        frame, depth = inspect.currentframe(), 0
        while frame and (depth == 0 or frame.f_code.co_filename == logging.__file__):
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())
logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)

if __name__ == "__main__":
    args = parameters.parser.parse_args()

    logger.add(
        sys.stderr,
        format="<cyan>{time:YYYY-MM-DD HH:mm:ss}</cyan> |"
        "<level>{level: ^8}</level>| "
        "<magenta>({name}:{line})</magenta> <level>{message}</level>",
        level=("DEBUG" if args.verbose > 0 else "INFO"),
        colorize=True,
        enqueue=True
    )

    execute(args)

