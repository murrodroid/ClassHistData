import os

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import pandas as pd
import datetime
import random
from tqdm import tqdm
from collections import defaultdict

from sklearn.preprocessing import *
from sklearn.model_selection import train_test_split

import nltk
from nltk import ngrams