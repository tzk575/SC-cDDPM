

import os
import logging
import numpy as np

import torch
import torch.nn as nn

from config import params


from utils.tools import setup_logging


logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")





def train(args):
    setup_logging(args.run_name)
    device = args.device




