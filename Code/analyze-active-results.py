'''generate-active-plots by Lucia Gordon'''

# imports
import re
import glob
import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from shutil import rmtree
from os import mkdir, path
from sys import argv

# global variables
labeling_budget = 500
batch_size = 10
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'tab:orange', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray']
marker = '.'
markersize = 5
modalities = ['a-tr-fused-r', 'a-tr-fused-u', 'a-tr-fused-c', 'a-tr-d', 'a-tr-fused-m', 'a-tr-m']

# functions
def get_accuracies(modality, start_trial, end_trial):
    return np.array([list(np.load('results/'+modality+'/accuracy-vs-labels-'+str(trial)+'.npy'))[:int(labeling_budget/batch_size)+1] for trial in range(start_trial, end_trial)])

def get_mean_accuracies(modality, start_trial, end_trial):
    return np.mean(get_accuracies(modality, start_trial, end_trial), axis=0)

def get_se_accuracies(modality, start_trial, end_trial):
    return np.std(get_accuracies(modality, start_trial, end_trial), axis=0)/np.sqrt(end_trial-start_trial)

def get_frac_middens_found(modality, start_trial, end_trial):
    return np.array([list(np.append(0, np.load('results/'+modality+'/fraction-middens-found'+str(trial)+'.npy')))[:int(labeling_budget/batch_size)+1] for trial in range(start_trial, end_trial)])

def get_mean_frac_middens_found(modality, start_trial, end_trial):
    return np.mean(get_frac_middens_found(modality, start_trial, end_trial), axis=0)

def get_se_frac_middens_found(modality, start_trial, end_trial):
    return np.std(get_frac_middens_found(modality, start_trial, end_trial), axis=0)/np.sqrt(end_trial-start_trial)

def two_sample_t_test(group1, group2): # the length of the groups should be the number of trials
    return stats.ttest_ind(a=group1, b=group2, equal_var=True)

# plot accuracies
mean_accuracies = []
mean_accuracies.append(get_mean_accuracies(modalities[0], 30, 60)) # random
mean_accuracies.append(get_mean_accuracies(modalities[1], 30, 60)) # uncertainty
mean_accuracies.append(get_mean_accuracies(modalities[2], 30, 60)) # positive certainty
mean_accuracies.append(get_mean_accuracies(modalities[3], 30, 60)) # disagree
mean_accuracies.append(get_mean_accuracies(modalities[4], 60, 90)) # multimodAL thermal+RGB fused
mean_accuracies.append(get_mean_accuracies(modalities[5], 60, 90)) # multimodAL thermal + RGB

se_accuracies = []
se_accuracies.append(get_se_accuracies(modalities[0], 30, 60)) # random
se_accuracies.append(get_se_accuracies(modalities[1], 30, 60)) # uncertainty
se_accuracies.append(get_se_accuracies(modalities[2], 30, 60)) # positive certainty
se_accuracies.append(get_se_accuracies(modalities[3], 30, 60)) # disagree
se_accuracies.append(get_se_accuracies(modalities[4], 60, 90)) # multimodAL thermal+RGB fused
se_accuracies.append(get_se_accuracies(modalities[5], 60, 90)) # multimodAL thermal + RGB

plt.figure('Accuracies for Active Learning Methods', dpi=300)
plt.plot(range(0, labeling_budget+1, batch_size), int(labeling_budget/batch_size+1)*[0.864], c=colors[0], label='Passive: T+R Fused', linestyle='dashed')
plt.plot(range(0, labeling_budget+1, batch_size), mean_accuracies[4], c=colors[1], label='MultimodAL: T+R Fused')
plt.errorbar(range(0, labeling_budget+1, batch_size), mean_accuracies[4], 1*np.array(se_accuracies[4]), marker=marker, markersize=markersize, color=colors[1])
plt.plot(range(0, labeling_budget+1, batch_size), mean_accuracies[5], c=colors[2], label='MultimodAL: T + R')
plt.errorbar(range(0, labeling_budget+1, batch_size), mean_accuracies[5], 1*np.array(se_accuracies[5]), marker=marker, markersize=markersize, color=colors[2])
plt.plot(range(0, labeling_budget+1, batch_size), mean_accuracies[2], c=colors[3], label='Positive Certainty: T+R Fused')
plt.errorbar(range(0, labeling_budget+1, batch_size), mean_accuracies[2], 1*np.array(se_accuracies[2]), marker=marker, markersize=markersize, color=colors[3])
plt.plot(range(0, labeling_budget+1, batch_size), mean_accuracies[3], c=colors[4], label='Disagree: T + R')
plt.errorbar(range(0, labeling_budget+1, batch_size), mean_accuracies[3], 1*np.array(se_accuracies[3]), marker=marker, markersize=markersize, color=colors[4])
plt.plot(range(0, labeling_budget+1, batch_size), mean_accuracies[1], c=colors[5], label='Uncertainty: T+R Fused')
plt.errorbar(range(0, labeling_budget+1, batch_size), mean_accuracies[1], 1*np.array(se_accuracies[1]), marker=marker, markersize=markersize, color=colors[5])
plt.plot(range(0, labeling_budget+1, batch_size), mean_accuracies[0], c=colors[6], label='Random: T+R Fused')
plt.errorbar(range(0, labeling_budget+1, batch_size), mean_accuracies[0], 1*np.array(se_accuracies[0]), marker=marker, markersize=markersize, color=colors[6])
plt.xlabel('Labeling Budget', fontsize=12)
plt.xticks(range(0, labeling_budget+1, 100), fontsize=12)
plt.ylabel('Test Accuracy', fontsize=12)
plt.yticks(np.arange(0.5, 1, step=0.1), fontsize=12)
plt.subplots_adjust(bottom=0.3)
plt.legend(fontsize=10, ncol=2)
plt.savefig('results/figs/active-learning-accuracies.png', bbox_inches='tight')

print('MultimodAL: T + R accuracy at 500 labels =', round(mean_accuracies[5][int(500/batch_size)],2))
print('MultimodAL: T + R accuracy error at 500 labels =', round(se_accuracies[5][int(500/batch_size)],2))

print('MultimodAL: T + R accuracy at 200 labels =', round(mean_accuracies[5][int(200/batch_size)],2))
print('MultimodAL: T + R accuracy error at 200 labels =', round(se_accuracies[5][int(200/batch_size)],2))

print('MultimodAL: T + R accuracy at 50 labels =', round(mean_accuracies[5][int(50/batch_size)],2))
print('MultimodAL: T + R accuracy error at 50 labels =', round(se_accuracies[5][int(50/batch_size)],2))

print(mean_accuracies[4][int(500/batch_size)])
print(mean_accuracies[4][int(500/batch_size)])
print(mean_accuracies[5][int(500/batch_size)])
print(se_accuracies[4][int(500/batch_size)])
print(se_accuracies[5][int(500/batch_size)])

# plot fraction middens found
mean_frac_middens_found = []
mean_frac_middens_found.append(get_mean_frac_middens_found(modalities[0], 30, 60)) # random
mean_frac_middens_found.append(get_mean_frac_middens_found(modalities[1], 30, 60)) # uncertainty
mean_frac_middens_found.append(get_mean_frac_middens_found(modalities[2], 30, 60)) # positive certainty
mean_frac_middens_found.append(get_mean_frac_middens_found(modalities[3], 30, 60)) # disagree
mean_frac_middens_found.append(get_mean_frac_middens_found(modalities[4], 60, 90)) # multimodAL thermal+RGB fused
mean_frac_middens_found.append(get_mean_frac_middens_found(modalities[5], 60, 90)) # multimodAL thermal + RGB

se_frac_middens_found = []
se_frac_middens_found.append(get_se_frac_middens_found(modalities[0], 30, 60)) # random
se_frac_middens_found.append(get_se_frac_middens_found(modalities[1], 30, 60)) # uncertainty
se_frac_middens_found.append(get_se_frac_middens_found(modalities[2], 30, 60)) # positive certainty
se_frac_middens_found.append(get_se_frac_middens_found(modalities[3], 30, 60)) # disagree
se_frac_middens_found.append(get_se_frac_middens_found(modalities[4], 60, 90)) # multimodAL thermal+RGB fused
se_frac_middens_found.append(get_se_frac_middens_found(modalities[5], 60, 90)) # multimodAL thermal + RGB

plt.figure('Fraction of Middens Found\nwith Active Learning Methods', dpi=300)
plt.plot(range(0, labeling_budget+1, batch_size), mean_frac_middens_found[4], c=colors[1], label='MultimodAL: T+R Fused')
plt.errorbar(range(0, labeling_budget+1, batch_size), mean_frac_middens_found[4], 1*np.array(se_frac_middens_found[4]), marker=marker, markersize=markersize, color=colors[1])
plt.plot(range(0, labeling_budget+1, batch_size), mean_frac_middens_found[5], c=colors[2], label='MultimodAL: T + R')
plt.errorbar(range(0, labeling_budget+1, batch_size), mean_frac_middens_found[5], 1*np.array(se_frac_middens_found[5]), marker=marker, markersize=markersize, color=colors[2])
plt.plot(range(0, labeling_budget+1, batch_size), mean_frac_middens_found[2], c=colors[3], label='Positive Certainty: T+R Fused')
plt.errorbar(range(0, labeling_budget+1, batch_size), mean_frac_middens_found[2], 1*np.array(se_frac_middens_found[2]), marker=marker, markersize=markersize, color=colors[3])
plt.plot(range(0, labeling_budget+1, batch_size), mean_frac_middens_found[3], c=colors[4], label='Disagree: T + R')
plt.errorbar(range(0, labeling_budget+1, batch_size), mean_frac_middens_found[3], 1*np.array(se_frac_middens_found[3]), marker=marker, markersize=markersize, color=colors[4])
plt.plot(range(0, labeling_budget+1, batch_size), mean_frac_middens_found[1], c=colors[5], label='Uncertainty: T+R Fused')
plt.errorbar(range(0, labeling_budget+1, batch_size), mean_frac_middens_found[1], 1*np.array(se_frac_middens_found[1]), marker=marker, markersize=markersize, color=colors[5])
plt.plot(range(0, labeling_budget+1, batch_size), mean_frac_middens_found[0], c=colors[6], label='Random: T+R Fused')
plt.errorbar(range(0, labeling_budget+1, batch_size), mean_frac_middens_found[0], 1*np.array(se_frac_middens_found[0]), marker=marker, markersize=markersize, color=colors[6])
plt.xlabel('Labeling Budget', fontsize=12)
plt.xticks(range(0, labeling_budget+1, 100), fontsize=12)
plt.ylabel('Fraction of Training Middens Found', fontsize=12)
plt.yticks(np.arange(0, 0.8, step=0.1), fontsize=12)
plt.subplots_adjust(bottom=0.3)
plt.legend(fontsize=10, loc='upper left')
plt.savefig('results/figs/active-learning-fraction-middens-found.png', bbox_inches='tight')

print('Positive Certainty fraction of middens found at 100 labels =', mean_frac_middens_found[2][int(100/batch_size)])
print('MultimodAL: T+R Fused fraction of middens found at 100 labels =', mean_frac_middens_found[4][int(100/batch_size)])
print('MultimodAL: T + R fraction of middens found at 100 labels =', mean_frac_middens_found[5][int(100/batch_size)])
print('MultimodAL: T+R Fused fraction of middens found at 200 labels =', mean_frac_middens_found[4][int(200/batch_size)])
print('MultimodAL: T + R fraction of middens found at 200 labels =', mean_frac_middens_found[5][int(200/batch_size)])
print('MultimodAL: T+R Fused fraction of middens found at 500 labels =', mean_frac_middens_found[4][int(500/batch_size)])
print('MultimodAL: T + R fraction of middens found at 500 labels =', mean_frac_middens_found[5][int(500/batch_size)])

# statistical analysis
random_accuracies = get_accuracies(modalities[0], 30, 60).T[-1].T # random
uncertainty_accuracies = get_accuracies(modalities[1], 30, 60).T[-1].T # uncertainty
certainty_accuracies = get_accuracies(modalities[2], 30, 60).T[-1].T # positive certainty
disagree_accuracies = get_accuracies(modalities[3], 30, 60).T[-1].T # disagree
multimodAL_fused_accuracies = get_accuracies(modalities[4], 60, 90).T[-1].T # multimodAL thermal+RGB fused
multimodAL_accuracies = get_accuracies(modalities[5], 60, 90).T[-1].T # multimodAL thermal + RGB

# multimodAL thermal+RGB fused vs. baselines (all less than 0.05)
print(two_sample_t_test(multimodAL_fused_accuracies, random_accuracies))
print(two_sample_t_test(multimodAL_fused_accuracies, uncertainty_accuracies))
print(two_sample_t_test(multimodAL_fused_accuracies, certainty_accuracies))
print(two_sample_t_test(multimodAL_fused_accuracies, disagree_accuracies))

# multimodAL thermal + RGB vs. baselines (all less than 0.05)
print(two_sample_t_test(multimodAL_accuracies, random_accuracies))
print(two_sample_t_test(multimodAL_accuracies, uncertainty_accuracies))
print(two_sample_t_test(multimodAL_accuracies, certainty_accuracies))
print(two_sample_t_test(multimodAL_accuracies, disagree_accuracies))

# comparing baselines to random (not all less than 0.05)
print(two_sample_t_test(uncertainty_accuracies, random_accuracies))
print(two_sample_t_test(certainty_accuracies, random_accuracies))
print(two_sample_t_test(disagree_accuracies, random_accuracies))

passive_tr_fused_accuracies = [0.833, 0.861, 0.778, 0.889, 0.861, 0.917, 0.806, 0.889, 0.806, 0.861, 0.861, 0.833,
 0.889, 0.861, 0.806, 0.917, 0.917, 0.833, 0.806, 0.833, 0.944, 0.889, 0.833, 0.861,
 0.889, 0.889, 0.889, 0.833, 0.944, 0.889]

# comparing multimodAL to passive
print(two_sample_t_test(multimodAL_fused_accuracies, passive_tr_fused_accuracies))
print(two_sample_t_test(multimodAL_accuracies, passive_tr_fused_accuracies))