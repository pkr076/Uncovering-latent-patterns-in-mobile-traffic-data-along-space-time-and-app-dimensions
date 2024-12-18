# %%
import os
import sys
import numpy as np
import pandas as pd
import tensorly as tl
import time
import copy
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colrs

import pickle as pkl

from tqdm import tqdm
import torch
import gc

# %%
tl.set_backend('pytorch')

os.environ['CUDA_DEVICE_ORDER']= 'PCI_BUS_ID' # see issue #152
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# %%
USER = os.getlogin()
WORKING_DIR = f'/home/{USER}/data/Land_use'
DATA_DIR = f'{WORKING_DIR}/data'
IMAGE_DIR = f'{WORKING_DIR}/images'
CITY_NAME = 'Paris'

# %%
rank = [5, 5, 5]

if len(sys.argv) > 4:
    if sys.argv[1] == '-my_params':
    
        rank = [int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])]

# %%
fd = open(f'{DATA_DIR}/{CITY_NAME}/{CITY_NAME}_iris_srca_tensor.pkl', 'rb')
numpy_tensor = pkl.load(fd)
numpy_tensor = numpy_tensor[0]
fd.close()

# %%
start_time = time.time()

tensorly_tensor = tl.tensor(numpy_tensor, device='cuda', dtype=torch.float32)
core, factors = tl.decomposition.tucker(tensorly_tensor, rank=rank)
end_time = time.time()
delta_time = (end_time - start_time)

tensorly_tensor_approx = tl.tucker_to_tensor((core, factors))
mse_tensor = tl.metrics.regression.MSE(tensorly_tensor, tensorly_tensor_approx)
rmse_tensor = tl.metrics.regression.RMSE(tensorly_tensor, tensorly_tensor_approx)
# congruence, _ = tl.metrics.congruence_coefficient(tensorly_tensor, tensorly_tensor_approx)

numpy_tensor_approx = tensorly_tensor_approx.cpu().numpy()
l2_norm_error = np.linalg.norm(numpy_tensor - numpy_tensor_approx)

print(f'{rank[0]}\t{rank[1]}\t{rank[2]}\t{delta_time:.2f}\t{mse_tensor:.4f}\t{rmse_tensor:.4f}\t{l2_norm_error:.4f}')

# %%
time_factor = factors[2].cpu().numpy()

# %%
list_time = []
for i in range(0,48):
    hour = int(i/2)
    hour = f"{hour:02}" # add leading zero
    if i%2 == 0:
        minute = '00'
    else:
        minute = '30'
    list_time.append(f'{hour}:{minute}')

# %%
def save_time_factors(factor_matrix, list_ranks):   # tuple_ranks = (space, app, time)
    global_min_value = np.min(factor_matrix)
    global_max_value = np.max(factor_matrix)
    #print(global_min_value, global_max_value)

    my_cmap = plt.cm.RdBu_r
    my_norm = plt.cm.colors.TwoSlopeNorm(vmin=global_min_value, vmax=global_max_value/2, vcenter=0)

    labels = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
    fig = plt.figure(figsize=(20,40))
    #fig.suptitle(f'TimeFactor_rank_545')
    list_axes = []

    for i in range(0, list_ranks[2]):
        #time_factor_i = time_factor_loaded[:,i]
        time_factor_i = time_factor[:,i]
        reshaped_time_factor_i = time_factor_i.reshape(7,48)
        labels = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
        x = list(range(0,48))
        
        ax = fig.add_subplot(list_ranks[2], 1, i+1)
        list_axes.append(ax)
        ax.imshow(reshaped_time_factor_i, cmap=my_cmap, interpolation='nearest', aspect='auto', norm = my_norm)
        ax.set_xticks(x)
        ax.set_xticklabels(list_time, rotation=90)
        ax.set_yticks(range(0,7))
        ax.set_yticklabels(labels)
        ax.set_title(f'Time Factor_{i}_rank_{list_ranks[0]}{list_ranks[1]}{list_ranks[2]}')

    plt.colorbar(plt.cm.ScalarMappable(cmap=my_cmap, norm= my_norm), ax=list_axes)
    plt.savefig(f'{IMAGE_DIR}/{CITY_NAME}_time_factors/{CITY_NAME}_{rank[0]}{rank[1]}{rank[2]}.png', bbox_inches='tight', dpi=300)
    plt.close()

# %%
save_time_factors(time_factor, rank)


