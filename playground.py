import torch
import os
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from numpngw import write_apng
from IPython.display import Image
from tqdm.notebook import tqdm

from utils.panda_pushing_env import PandaImageSpacePushingEnv
from utils.visualizers import GIFVisualizer, NotebookVisualizer
from utils.utils import *
from sindy.SINDy import *

test_mat = torch.ones((64, 1, 32, 32), dtype=torch.float32)
model = RoboSINDy(input_dim=1*32*32, batch_size=64)

print("Model initialized successfully.")

x, z, x_hat, z_next = model(test_mat)

print(f"x: {x.shape}, z: {z.shape}, x_hat: {x_hat.shape}, z_next: {z_next.shape}")