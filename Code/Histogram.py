from numpy import load, histogram, save, nonzero, array, max, where
from matplotlib.pyplot import figure, scatter, plot, xlabel, ylabel, xticks, legend, title, savefig, show, ylim, hist

middenImages = load('Phase3/Data/Thermal/RawThermalMiddenImages.npy')
emptyImages = load('Phase3/Data/Thermal/RawThermalEmptyImages.npy')

maxOccupiedBinMiddens = []

for midden in middenImages:
    histogramData, binEdges = histogram(midden, bins=100, range=(0,797))
    maxOccupiedBinMiddens.append(max(nonzero(histogramData)))

save('Phase3/Data/Thermal/MaxOccupiedBinMiddens', maxOccupiedBinMiddens)

maxOccupiedBinEmpty = []

for empty in emptyImages:
    histogramData, binEdges = histogram(empty, bins=100, range=(0,797))
    maxOccupiedBinEmpty.append(max(nonzero(histogramData)))

save('Phase3/Data/Thermal/MaxOccupiedBinEmpty', maxOccupiedBinEmpty)

def numMiddensWithBinOverN(n):
    return len(where(array(maxOccupiedBinMiddens)>=n)[0])

def numEmptyWithBinOverN(n):
    return len(where(array(maxOccupiedBinEmpty)>=n)[0])

def probBinOverN(n):
    return (numMiddensWithBinOverN(n) + numEmptyWithBinOverN(n))/(len(maxOccupiedBinMiddens)+len(maxOccupiedBinEmpty))

probMidden = len(maxOccupiedBinMiddens)/(len(maxOccupiedBinMiddens)+len(maxOccupiedBinEmpty))
numMiddens = len(maxOccupiedBinMiddens)

def probMiddenGivenBinOverN(n):
    return numMiddensWithBinOverN(n) * probMidden / (probBinOverN(n) * numMiddens)

figure('Midden Probability vs. Max Occupied Bin Threshold', dpi=300)
title('Midden Probability vs. Max Occupied Bin Threshold')
xlabel('Threshold')
ylabel('Midden Probability')
plot(range(100), [probMiddenGivenBinOverN(n) for n in range(100)])
savefig('Phase3/Histograms/MiddenProbability.png')