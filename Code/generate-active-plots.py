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
labeling_budget = 800
batch_size = 10
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'tab:orange', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray']
marker = '.'
markersize = 5

modalities = ['a-tr-fused-r', 'a-tr-fused-u', 'a-tr-fused-c', 'a-tr-d', 'a-tr-fused-v', 'a-t-m', 'a-tr-fused-m', 'a-tr-m', 'a-tr-fused-a']

# functions
def get_accuracies(modality, start_trial, end_trial):
    return np.array([list(np.load('results/'+modality+'/accuracy-vs-labels-'+str(trial)+'.npy')) for trial in range(start_trial, end_trial)])

def get_mean_accuracies(modality, start_trial, end_trial):
    return np.mean(get_accuracies(modality, start_trial, end_trial), axis=0)

def get_se_accuracies(modality, start_trial, end_trial):
    return np.std(get_accuracies(modality, start_trial, end_trial), axis=0)/np.sqrt(end_trial-start_trial)

def get_frac_middens_found(modality, start_trial, end_trial):
    return np.array([list(np.append(0, np.load('results/'+modality+'/fraction-middens-found'+str(trial)+'.npy'))) for trial in range(start_trial, end_trial)])

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
mean_accuracies.append(get_mean_accuracies(modalities[4], 60, 90)) # rule violation
mean_accuracies.append(get_mean_accuracies(modalities[5], 60, 90)) # multimodAL thermal
mean_accuracies.append(get_mean_accuracies(modalities[6], 60, 90)) # multimodAL thermal+RGB fused
mean_accuracies.append(get_mean_accuracies(modalities[7], 60, 90)) # multimodAL thermal + RGB
mean_accuracies.append(get_mean_accuracies(modalities[8], 30, 60)) # ablation

se_accuracies = []
se_accuracies.append(get_se_accuracies(modalities[0], 30, 60)) # random
se_accuracies.append(get_se_accuracies(modalities[1], 30, 60)) # uncertainty
se_accuracies.append(get_se_accuracies(modalities[2], 30, 60)) # positive certainty
se_accuracies.append(get_se_accuracies(modalities[3], 30, 60)) # disagree
se_accuracies.append(get_se_accuracies(modalities[4], 60, 90)) # rule violation
se_accuracies.append(get_se_accuracies(modalities[5], 60, 90)) # multimodAL thermal
se_accuracies.append(get_se_accuracies(modalities[6], 60, 90)) # multimodAL thermal+RGB fused
se_accuracies.append(get_se_accuracies(modalities[7], 60, 90)) # multimodAL thermal + RGB
se_accuracies.append(get_se_accuracies(modalities[8], 30, 60)) # ablation

plt.figure('Accuracies for Active Learning Methods', dpi=300)
plt.title('Accuracies for Active Learning Methods', fontsize=14)
plt.plot(range(0, labeling_budget+1, batch_size), mean_accuracies[0], c=colors[0], label='Random: T+R Fused')
plt.errorbar(range(0, labeling_budget+1, batch_size), mean_accuracies[0], 1*np.array(se_accuracies[0]), marker=marker, markersize=markersize, color=colors[0])
plt.plot(range(0, labeling_budget+1, batch_size), mean_accuracies[1], c=colors[1], label='Uncertainty: T+R Fused')
plt.errorbar(range(0, labeling_budget+1, batch_size), mean_accuracies[1], 1*np.array(se_accuracies[1]), marker=marker, markersize=markersize, color=colors[1])
plt.plot(range(0, labeling_budget+1, batch_size), mean_accuracies[2], c=colors[2], label='Positive Certainty: T+R Fused')
plt.errorbar(range(0, labeling_budget+1, batch_size), mean_accuracies[2], 1*np.array(se_accuracies[2]), marker=marker, markersize=markersize, color=colors[2])
plt.plot(range(0, labeling_budget+1, batch_size), mean_accuracies[3], c=colors[3], label='Disagree: T + R')
plt.errorbar(range(0, labeling_budget+1, batch_size), mean_accuracies[3], 1*np.array(se_accuracies[3]), marker=marker, markersize=markersize, color=colors[3])
plt.plot(range(0, labeling_budget+1, batch_size), mean_accuracies[4], c=colors[4], label='Rule Violation: T+R Fused')
plt.errorbar(range(0, labeling_budget+1, batch_size), mean_accuracies[4], 1*np.array(se_accuracies[4]), marker=marker, markersize=markersize, color=colors[4])
plt.plot(range(0, labeling_budget+1, batch_size), mean_accuracies[5], c=colors[5], label='MultimodAL: T')
plt.errorbar(range(0, labeling_budget+1, batch_size), mean_accuracies[5], 1*np.array(se_accuracies[5]), marker=marker, markersize=markersize, color=colors[5])
plt.plot(range(0, labeling_budget+1, batch_size), mean_accuracies[6], c=colors[6], label='MultimodAL: T+R Fused')
plt.errorbar(range(0, labeling_budget+1, batch_size), mean_accuracies[6], 1*np.array(se_accuracies[6]), marker=marker, markersize=markersize, color=colors[6])
plt.plot(range(0, labeling_budget+1, batch_size), mean_accuracies[7], c=colors[7], label='MultimodAL: T + R')
plt.errorbar(range(0, labeling_budget+1, batch_size), mean_accuracies[7], 1*np.array(se_accuracies[7]), marker=marker, markersize=markersize, color=colors[7])
plt.plot(range(0, labeling_budget+1, batch_size), mean_accuracies[8], c=colors[8], label='Ablation')
plt.errorbar(range(0, labeling_budget+1, batch_size), mean_accuracies[8], 1*np.array(se_accuracies[8]), marker=marker, markersize=markersize, color=colors[8])
plt.plot(range(0, labeling_budget+1, batch_size), int(labeling_budget/batch_size+1)*[0.864], c=colors[9], label='Passive: T+R Fused', linestyle='dashed')
plt.xlabel('# Images Labeled', fontsize=12)
plt.xticks(range(0, labeling_budget+1, 100), fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.yticks(np.arange(0.5, 1, step=0.1), fontsize=12)
plt.subplots_adjust(bottom=0.3)
plt.legend(fontsize=10, ncol=2)
plt.savefig('results/figs/active-learning-accuracies.png', bbox_inches='tight')

# plot fraction middens found
mean_frac_middens_found = []
mean_frac_middens_found.append(get_mean_frac_middens_found(modalities[0], 30, 60)) # random
mean_frac_middens_found.append(get_mean_frac_middens_found(modalities[1], 30, 60)) # uncertainty
mean_frac_middens_found.append(get_mean_frac_middens_found(modalities[2], 30, 60)) # positive certainty
mean_frac_middens_found.append(get_mean_frac_middens_found(modalities[3], 30, 60)) # disagree
mean_frac_middens_found.append(get_mean_frac_middens_found(modalities[4], 60, 90)) # rule violation
mean_frac_middens_found.append(get_mean_frac_middens_found(modalities[5], 60, 90)) # multimodAL thermal
mean_frac_middens_found.append(get_mean_frac_middens_found(modalities[6], 60, 90)) # multimodAL thermal+RGB fused
mean_frac_middens_found.append(get_mean_frac_middens_found(modalities[7], 60, 90)) # multimodAL thermal + RGB
mean_frac_middens_found.append(get_mean_frac_middens_found(modalities[8], 30, 60)) # ablation

se_frac_middens_found = []
se_frac_middens_found.append(get_se_frac_middens_found(modalities[0], 30, 60)) # random
se_frac_middens_found.append(get_se_frac_middens_found(modalities[1], 30, 60)) # uncertainty
se_frac_middens_found.append(get_se_frac_middens_found(modalities[2], 30, 60)) # positive certainty
se_frac_middens_found.append(get_se_frac_middens_found(modalities[3], 30, 60)) # disagree
se_frac_middens_found.append(get_se_frac_middens_found(modalities[4], 60, 90)) # rule violation
se_frac_middens_found.append(get_se_frac_middens_found(modalities[5], 60, 90)) # multimodAL thermal
se_frac_middens_found.append(get_se_frac_middens_found(modalities[6], 60, 90)) # multimodAL thermal+RGB fused
se_frac_middens_found.append(get_se_frac_middens_found(modalities[7], 60, 90)) # multimodAL thermal + RGB
se_frac_middens_found.append(get_se_frac_middens_found(modalities[8], 30, 60)) # ablation

plt.figure('Fraction of Middens Found\nwith Active Learning Methods', dpi=300)
plt.title('Fraction of Middens Found\nwith Active Learning Methods', fontsize=14)
plt.plot(range(0, labeling_budget+1, batch_size), mean_frac_middens_found[0], c=colors[0], label='Random: T+R Fused')
plt.errorbar(range(0, labeling_budget+1, batch_size), mean_frac_middens_found[0], 1*np.array(se_frac_middens_found[0]), marker=marker, markersize=markersize, color=colors[0])
plt.plot(range(0, labeling_budget+1, batch_size), mean_frac_middens_found[1], c=colors[1], label='Uncertainty: T+R Fused')
plt.errorbar(range(0, labeling_budget+1, batch_size), mean_frac_middens_found[1], 1*np.array(se_frac_middens_found[1]), marker=marker, markersize=markersize, color=colors[1])
plt.plot(range(0, labeling_budget+1, batch_size), mean_frac_middens_found[2], c=colors[2], label='Positive Certainty: T+R Fused')
plt.errorbar(range(0, labeling_budget+1, batch_size), mean_frac_middens_found[2], 1*np.array(se_frac_middens_found[2]), marker=marker, markersize=markersize, color=colors[2])
plt.plot(range(0, labeling_budget+1, batch_size), mean_frac_middens_found[3], c=colors[3], label='Disagree: T + R')
plt.errorbar(range(0, labeling_budget+1, batch_size), mean_frac_middens_found[3], 1*np.array(se_frac_middens_found[3]), marker=marker, markersize=markersize, color=colors[3])
plt.plot(range(0, labeling_budget+1, batch_size), mean_frac_middens_found[4], c=colors[4], label='Rule Violation: T+R Fused')
plt.errorbar(range(0, labeling_budget+1, batch_size), mean_frac_middens_found[4], 1*np.array(se_frac_middens_found[4]), marker=marker, markersize=markersize, color=colors[4])
plt.plot(range(0, labeling_budget+1, batch_size), mean_frac_middens_found[5], c=colors[5], label='MultimodAL: T')
plt.errorbar(range(0, labeling_budget+1, batch_size), mean_frac_middens_found[5], 1*np.array(se_frac_middens_found[5]), marker=marker, markersize=markersize, color=colors[5])
plt.plot(range(0, labeling_budget+1, batch_size), mean_frac_middens_found[6], c=colors[6], label='MultimodAL: T+R Fused')
plt.errorbar(range(0, labeling_budget+1, batch_size), mean_frac_middens_found[6], 1*np.array(se_frac_middens_found[6]), marker=marker, markersize=markersize, color=colors[6])
plt.plot(range(0, labeling_budget+1, batch_size), mean_frac_middens_found[7], c=colors[7], label='MultimodAL: T + R')
plt.errorbar(range(0, labeling_budget+1, batch_size), mean_frac_middens_found[7], 1*np.array(se_frac_middens_found[7]), marker=marker, markersize=markersize, color=colors[7])
plt.plot(range(0, labeling_budget+1, batch_size), mean_frac_middens_found[8], c=colors[8], label='Ablation')
plt.errorbar(range(0, labeling_budget+1, batch_size), mean_frac_middens_found[8], 1*np.array(se_frac_middens_found[8]), marker=marker, markersize=markersize, color=colors[8])
plt.xlabel('# Images Labeled', fontsize=12)
plt.xticks(range(0, labeling_budget+1, 100), fontsize=12)
plt.ylabel('Fraction of Middens Found', fontsize=12)
plt.yticks(np.arange(0, 0.9, step=0.1), fontsize=12)
plt.subplots_adjust(bottom=0.4)
plt.legend(fontsize=10, loc='upper left')
plt.savefig('results/figs/active-learning-fraction-middens-found.png', bbox_inches='tight')


# # print('MultimodAL: T + R')
# # print('50 labels acc:', mean_accuracies[5][int(50/batch_size)])
# # print('50 labels err:', standard_errors[5][int(50/batch_size)])
# # print('250 labels acc:', mean_accuracies[5][int(250/batch_size)])
# # print('250 labels err:', standard_errors[5][int(250/batch_size)])
# # print('500 labels acc:', mean_accuracies[5][int(500/batch_size)])
# # print('500 labels err:', standard_errors[5][int(500/batch_size)])


# statistical analysis

# ablation vs. TR-fused
tr_fused_accuracies = get_accuracies(modalities[6], 60, 90).T[-1].T
print(tr_fused_accuracies.shape)
ablation_accuracies = get_accuracies(modalities[8], 30, 60).T[-1].T
print(two_sample_t_test(tr_fused_accuracies, ablation_accuracies))