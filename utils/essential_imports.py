#### Ignore Warnings ####
import warnings
### Progress Bar ###
from tqdm.notebook import tqdm


import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler #For scaling Dataset
from matplotlib.colors import LinearSegmentedColormap #colorbars
from matplotlib.pyplot import cm #colormaps
from scipy.signal import savgol_filter # For smoothing some curves in plots

from tqdm import tqdm_notebook # progress bar


import torch
print('GPU and CUDA activated:', torch.cuda.is_available())
if torch.cuda.is_available():
    dev = "cuda:0" 
else: 
    dev = "cpu" 
dev="cpu"

device = torch.device(dev)

torch.set_default_dtype(torch.float64) #Float 64 default precision
