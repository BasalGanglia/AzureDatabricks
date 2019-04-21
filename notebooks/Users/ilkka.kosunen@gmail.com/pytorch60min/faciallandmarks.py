# Databricks notebook source
from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

plt.ion()   # interactive mode

# COMMAND ----------

have to import the data (later)
#landmarks_frame = pd.read_csv('data/faces/face_landmarks.csv')

# COMMAND ----------

