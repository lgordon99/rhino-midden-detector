'''generate-passive-plot by Lucia Gordon'''

# imports
import re
import glob
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from matplotlib.colors import ListedColormap
from shutil import rmtree
from os import mkdir, path
from sys import argv

# global variables
trials = 30
batch_size = 10
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'tab:orange', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray']
marker = '.'
markersize = 5

# functions
def hex_to_RGB(hex_str):
    """ #FFFFFF -> [255,255,255]"""
    return [int(hex_str[i:i+2], 16) for i in range(1,6,2)]

def get_color_gradient(c1, c2, n):
    assert n > 1
    c1_rgb = np.array(hex_to_RGB(c1))/255
    c2_rgb = np.array(hex_to_RGB(c2))/255
    mix_pcts = [x/(n-1) for x in range(n)]
    rgb_colors = [((1-mix)*c1_rgb + (mix*c2_rgb)) for mix in mix_pcts]

    for i in range(len(rgb_colors)):
        rgb_colors[i] = np.append(rgb_colors[i], 1)

    return rgb_colors

def gradient_image(ax, x, width, bottom, accuracy, cmap):
    extent = (x, x+width, bottom, accuracy)
    v = np.array([1, 0])
    X = np.array([[v @ [1, 0], v @ [1, 1]],[v @ [0, 0], v @ [0, 1]]])
    X = 1/X.max() * X
    im = ax.imshow(X, interpolation='bicubic', aspect='auto', cmap=cmap, extent=extent)
    return im

def two_sample_t_test(group1, group2): # the length of the groups should be the number of trials
    return stats.ttest_ind(a=group1, b=group2, equal_var=True)

# generate passive plot
modalities = ['p-tr-fused-', 'p-t-', 'p-tl-fused-', 'p-trl-fused-', 'p-r-', 'p-rl-fused-', 'p-l-']
accuracies = []
precisions = []
recalls = []
f1s =[]
standard_errors_acc = []
standard_errors_pre = []
standard_errors_rec = []
standard_errors_f1 = []

for modality in modalities:
    results = []

    for trial in range(trials):
        results.append(list(np.load('results/'+modality+'/results-'+str(trial)+'.npy')))

    results = np.array(results).T
    accuracy = np.mean(results[0])
    precision = np.mean(results[3])
    recall = np.mean(results[4])
    f1 = np.mean(results[5])

    standard_error_acc = np.std(results[0])/np.sqrt(trials)
    standard_error_pre = np.std(results[3])/np.sqrt(trials)
    standard_error_rec = np.std(results[4])/np.sqrt(trials)
    standard_error_f1 = np.std(results[5])/np.sqrt(trials)

    accuracies.append(accuracy)
    precisions.append(precision)
    recalls.append(recall)
    f1s.append(f1)

    standard_errors_acc.append(standard_error_acc)
    standard_errors_pre.append(standard_error_pre)
    standard_errors_rec.append(standard_error_rec)
    standard_errors_f1.append(standard_error_f1)

print('accuracies')
print(np.around(accuracies,3))
print(np.around(standard_errors_acc,3))
print('precisions')
print(np.around(precisions,3))
print(np.around(standard_errors_pre,3))
print('recalls')
print(np.around(recalls,3))
print(np.around(standard_errors_rec,3))
print('f1s')
print(f1s,3)
print(np.around(standard_errors_f1,3))

cyan = '#d95f02'
magenta = '#1b9e77'
yellow = '#7570b3'
b_gradient = ListedColormap(get_color_gradient(cyan, cyan, 100))
g_gradient = ListedColormap(get_color_gradient(magenta, magenta, 100))
r_gradient = ListedColormap(get_color_gradient(yellow, yellow, 100))
bg_gradient = ListedColormap(get_color_gradient(cyan, magenta, 100))
br_gradient = ListedColormap(get_color_gradient(cyan, yellow, 100))
gr_gradient = ListedColormap(get_color_gradient(magenta, yellow, 100))
bgr_gradient = ListedColormap(get_color_gradient(cyan, magenta, 50)+get_color_gradient(magenta, yellow, 50))

fig, ax = plt.subplots(dpi=300)
N = len(accuracies)
width = 0.1
bottom = 0.5
x = 1/(2*N)*np.arange(1,2*N,2)-width/2

# ax.set_title('Accuracies for Passive Learning Methods', fontsize=14)
ax.set(xlim=(0, 1), ylim=(bottom, 0.9), xmargin=0, xticks=x + width/2, xticklabels=['Thermal\n+RGB\nFused', 'Thermal', 'Thermal\n+LiDAR\nFused', 'Thermal\n+RGB\n+LiDAR\nFused', 'RGB', 'RGB\n+LiDAR\nFused', 'LiDAR'], yticks=np.arange(bottom,1,0.1))
ax.set_ylabel('Accuracy', fontsize=12)
gradient_image(ax, x[0], width, bottom, accuracies[0], cmap=bg_gradient)
plt.errorbar(x[0]+width/2, accuracies[0], 1*np.array(standard_errors_acc[0]), marker=marker, markersize=markersize, color=colors[0])
gradient_image(ax, x[1], width, bottom, accuracies[1], cmap=b_gradient)
plt.errorbar(x[1]+width/2, accuracies[1], 1*np.array(standard_errors_acc[1]), marker=marker, markersize=markersize, color=colors[0])
gradient_image(ax, x[2], width, bottom, accuracies[2], cmap=br_gradient)
plt.errorbar(x[2]+width/2, accuracies[2], 1*np.array(standard_errors_acc[2]), marker=marker, markersize=markersize, color=colors[0])
gradient_image(ax, x[3], width, bottom, accuracies[3], cmap=bgr_gradient)
plt.errorbar(x[3]+width/2, accuracies[3], 1*np.array(standard_errors_acc[3]), marker=marker, markersize=markersize, color=colors[0])
gradient_image(ax, x[4], width, bottom, accuracies[4], cmap=g_gradient)
plt.errorbar(x[4]+width/2, accuracies[4], 1*np.array(standard_errors_acc[4]), marker=marker, markersize=markersize, color=colors[0])
gradient_image(ax, x[5], width, bottom, accuracies[5], cmap=gr_gradient)
plt.errorbar(x[5]+width/2, accuracies[5], 1*np.array(standard_errors_acc[5]), marker=marker, markersize=markersize, color=colors[0])
gradient_image(ax, x[6], width, bottom, accuracies[6], cmap=r_gradient)
plt.errorbar(x[6]+width/2, accuracies[6], 1*np.array(standard_errors_acc[6]), marker=marker, markersize=markersize, color=colors[0])
plt.subplots_adjust(bottom=0.6)
plt.savefig('results/figs/passive-learning-accuracies.png', bbox_inches='tight')

def get_accuracies(modality, start_trial, end_trial):
    return np.array([list(np.load('results/'+modality+'/results-'+str(trial)+'.npy')) for trial in range(start_trial, end_trial)])

# thermal vs. thermal-RGB fused
tr_fused_accuracies = get_accuracies(modalities[0], 0, 30).T[0].T
print(tr_fused_accuracies)
thermal_accuracies = get_accuracies(modalities[1], 0, 30).T[0].T
print(two_sample_t_test(tr_fused_accuracies, thermal_accuracies))