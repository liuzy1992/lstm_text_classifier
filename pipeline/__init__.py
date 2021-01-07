#!/usr/bin/env python3

import torch
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

from .preprocessing import preprocessing
from .todataset import todataset
from .iterator import iterator
from .training import training
from .evaluation import evaluation
