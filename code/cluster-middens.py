'''cluster-middens by Lucia Gordon'''

import numpy as np
import scipy.cluster.vq as vq
import matplotlib.pyplot as plt
from sys import argv

folder = argv[1]
features = np.genfromtxt('data/midden-coordinates-m.csv', delimiter=',')[1:]
whitened = vq.whiten(features)
clusters = 6
codebook, distortion = vq.kmeans(whitened, clusters)

plt.figure('Midden Map', dpi=300)
plt.tick_params(left = False, right = False, labelleft = False, labelbottom = False, bottom = False)
plt.scatter(whitened[:, 0], whitened[:, 1], c='b', sizes=len(features)*[100])
plt.scatter(codebook[:, 0], codebook[:, 1], c='r', sizes=clusters*[400], marker='*')
plt.savefig(folder+'/results/figs/clustered-middens.png', bbox_inches='tight')