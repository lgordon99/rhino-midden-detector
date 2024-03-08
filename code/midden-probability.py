'''midden-probability by Lucia Gordon'''

import numpy as np
import matplotlib.pyplot as plt
from sys import argv

folder = argv[1]
labels = np.load('data/labels.npy')
midden_indices = np.where(labels == 1)[0]
empty_indices = np.where(labels == 0)[0]

max_pixel_vals = np.load('data/max-pixel-vals.npy')
max_pixel_vals_middens = np.take(max_pixel_vals, midden_indices)
max_pixel_vals_empty = np.take(max_pixel_vals, empty_indices)
optimal_value = np.amax(max_pixel_vals)

def num_middens_with_MPV_over_n(n):
    return len(np.where(np.array(max_pixel_vals_middens)>=n)[0])

def num_empty_with_MPV_over_n(n):
    return len(np.where(np.array(max_pixel_vals_empty)>=n)[0])

def prob_MPV_over_n(n):
    return (num_middens_with_MPV_over_n(n) + num_empty_with_MPV_over_n(n))/(len(max_pixel_vals_middens)+len(max_pixel_vals_empty))

prob_midden = len(max_pixel_vals_middens)/(len(max_pixel_vals_middens)+len(max_pixel_vals_empty))
num_middens = len(max_pixel_vals_middens)

def prob_midden_given_MPV_over_n(n):
    return num_middens_with_MPV_over_n(n) * prob_midden / (prob_MPV_over_n(n) * num_middens)

plt.figure('Midden Probability vs. Maximum Pixel Value Threshold', dpi=300)
plt.xlabel('Maximum Pixel Value Threshold', fontsize=12)
plt.ylabel('Midden Probability', fontsize=12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.subplots_adjust(left=0.4, bottom=0.5)
plt.plot(range(int(np.amax(max_pixel_vals_middens))), [prob_midden_given_MPV_over_n(n) for n in range(int(np.amax(max_pixel_vals_middens)))], c='b')
plt.vlines(optimal_value, 0, 1, colors='g', linestyles='dashed', label='target value')
plt.legend(fontsize=12)
plt.savefig(folder+'/results/figs/midden-probability.png', bbox_inches='tight')