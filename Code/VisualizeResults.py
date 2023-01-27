from numpy import array, mean, std, save, load
from shutil import rmtree
from os import mkdir, path
from matplotlib.pyplot import figure, scatter, plot, xlabel, ylabel, xticks, legend, title, savefig, show
from sys import argv
import re

numTrials = 30
epochs = 30
colors = ['lightcoral', 'indianred', 'brown', 'firebrick', 'maroon']

# Functions
def plotAccuraciesOverEpochs(setting):
    splitSetting = re.split('(?<=.)(?=[A-Z])', setting)
    if len(splitSetting) == 2:
        plotTitle = 'Accuracy vs. Epochs for the ' + splitSetting[0] + ' ' + splitSetting[1] + ' Model'
    elif len(splitSetting) == 4:
        plotTitle = 'Accuracy vs. Epochs for the ' + splitSetting[0] + ' ' + splitSetting[1] + splitSetting[2] + splitSetting[3] + ' Model'
    accuraciesOverEpochs = []

    for trial in range(1, numTrials+1):
        accuraciesOverEpochs.append(list(load('Phase3/Results/'+setting+'/AccuracyOverEpochs'+str(trial)+'.npy')))

    avgs = mean(accuraciesOverEpochs, axis=0)

    figure(plotTitle, dpi=300)
    title(plotTitle)

    for trial in range(1, numTrials+1):
        # scatter(range(1, epochs+1), accuraciesOverEpochs[trial-1], c=colors[trial-1])
        plot(range(1, epochs+1), accuraciesOverEpochs[trial-1], c=colors[trial-1], label='Trial '+str(trial))
        
    # scatter(range(1, epochs+1), avgs, c='b')
    plot(range(1, epochs+1), avgs, c='b', label='mean')
    legend()
    xlabel('Epochs')
    xticks(range(1,epochs+1))
    ylabel('Accuracy')
    savefig('Phase3/Results/'+setting+'/AccuracyOverEpochs.png')

def compareAccuraciesOverEpochs():
    thermalAccuracies = []
    RGBAccuracies = []
    LiDARAccuracies = []
    fusedAccuracies = []

    for trial in range(1, numTrials+1):
        thermalAccuracies.append(list(load('Phase3/Results/PassiveThermal/AccuracyOverEpochs'+str(trial)+'.npy')))
        RGBAccuracies.append(list(load('Phase3/Results/PassiveRGB/AccuracyOverEpochs'+str(trial)+'.npy')))
        LiDARAccuracies.append(list(load('Phase3/Results/PassiveLiDAR/AccuracyOverEpochs'+str(trial)+'.npy')))
        fusedAccuracies.append(list(load('Phase3/Results/PassiveFused/AccuracyOverEpochs'+str(trial)+'.npy')))

    thermalAvg = mean(thermalAccuracies, axis=0)
    RGBAvg = mean(RGBAccuracies, axis=0)
    LiDARAvg = mean(LiDARAccuracies, axis=0)
    fusedAvg = mean(fusedAccuracies, axis=0)

    figure('Accuracy vs. Epochs for Passive Learning', dpi=300)
    title('Accuracy vs. Epochs for Passive Learning Methods\nAveraged Over ' + str(numTrials) + ' Trials')
    plot(range(1, epochs+1), thermalAvg, c='b', label='Thermal')
    plot(range(1, epochs+1), RGBAvg, c='g', label='RGB')
    plot(range(1, epochs+1), LiDARAvg, c='r', label='LiDAR')
    plot(range(1, epochs+1), fusedAvg, c='c', label='Fused Thermal+RGB')
    legend()
    xlabel('Epochs')
    xticks(range(0,epochs+1,2))
    ylabel('Accuracy')
    savefig('Phase3/Results/PassiveLearning.png')

def plotAccuraciesVsImagesLabeled(setting):
    splitSetting = re.split('(?<=.)(?=[A-Z])', setting)
    print(splitSetting)
    plotTitle = 'Accuracy vs. Images Labeled for the ' + splitSetting[0] + ' ' + splitSetting[1] + ' Model'
    accuraciesVsImagesLabeled = []

    for trial in range(1, numTrials+1):
        accuraciesVsImagesLabeled.append(list(load('Phase3/Results/'+setting+'/AccuracyVsImagesLabeled'+str(trial)+'.npy')))
    
    figure(plotTitle, dpi=300)
    title(plotTitle)

    for trial in range(1, numTrials+1):
        scatter(range(16, 241, 16), accuraciesVsImagesLabeled[trial-1], c=colors[trial-1])
        plot(range(16, 241, 16), accuraciesVsImagesLabeled[trial-1], c=colors[trial-1])

    xlabel('Images Labeled')
    xticks(range(16, 241, 16))
    ylabel('Accuracy')
    savefig('Phase3/Results/'+setting+'/AccuracyVsImagesLabeled.png')

def compareAccuraciesVsImagesLabeled():
    plotTitle = 'Accuracy vs. # Images Labeled for Active Learning Methods\nAveraged Over ' + str(numTrials) + ' Trials'
    activeUncertaintyAccuracies = []
    activeCertaintyAccuracies = []
    activeBinningThermalAccuracies = []
    activeBinningFusedAccuracies = []
    activeDisagreeAccuracies = []
    activeBinningThermalRGBAccuracies = []
    activeBinningThermalRGBLiDARAccuracies = []

    for trial in range(1, numTrials+1):
        activeUncertaintyAccuracies.append(list(load('Phase3/Results/ActiveUncertainty/AccuracyVsImagesLabeled'+str(trial)+'.npy')))
        activeCertaintyAccuracies.append(list(load('Phase3/Results/ActiveCertainty/AccuracyVsImagesLabeled'+str(trial)+'.npy')))
        activeBinningThermalAccuracies.append(list(load('Phase3/Results/ActiveBinningThermal/AccuracyVsImagesLabeled'+str(trial)+'.npy')))
        activeBinningFusedAccuracies.append(list(load('Phase3/Results/ActiveBinningFused/AccuracyVsImagesLabeled'+str(trial)+'.npy')))
        activeDisagreeAccuracies.append(list(load('Phase3/Results/ActiveDisagree/AccuracyVsImagesLabeled'+str(trial)+'.npy')))
        activeBinningThermalRGBAccuracies.append(list(load('Phase3/Results/ActiveBinningThermalRGB/AccuracyVsImagesLabeled'+str(trial)+'.npy')))
        activeBinningThermalRGBLiDARAccuracies.append(list(load('Phase3/Results/ActiveBinningThermalRGBLiDAR/AccuracyVsImagesLabeled'+str(trial)+'.npy')))
    
    uncertaintyAvg = mean(activeUncertaintyAccuracies, axis=0)
    certaintyAvg = mean(activeCertaintyAccuracies, axis=0)
    binningThermalAvg = mean(activeBinningThermalAccuracies, axis=0)
    binningFusedAvg = mean(activeBinningFusedAccuracies, axis=0)
    disagreeAvg = mean(activeDisagreeAccuracies, axis=0)
    binningThermalRGBAvg = mean(activeBinningThermalRGBAccuracies, axis=0)
    binningThermalRGBLiDARAvg = mean(activeBinningThermalRGBLiDARAccuracies, axis=0)

    figure('Accuracy vs. # Images Labeled for Active Learning Methods\nAveraged Over ' + str(numTrials) + ' Trials', dpi=300)
    title(plotTitle)
    plot(range(10, 251, 10), uncertaintyAvg, c='b', label='Uncertainty – Thermal')
    plot(range(10, 251, 10), certaintyAvg, c='g', label='Certainty – Thermal')
    plot(range(10, 251, 10), binningThermalAvg, c='r', label='Binning – Thermal')
    plot(range(10, 251, 10), binningFusedAvg, c='c', label='Binning – Fused Thermal+RGB')
    plot(range(10, 251, 10), disagreeAvg, c='m', label='Thermal+RGB Disagree')
    plot(range(10, 251, 10), binningThermalRGBAvg, c='y', label='Binning – Thermal+RGB')
    plot(range(10, 251, 10), binningThermalRGBLiDARAvg, c='tab:blue', label='Binning – Thermal+RGB+LiDAR')
    legend()
    xlabel('# Images Labeled')
    xticks(range(10, 251, 20))
    ylabel('Accuracy')
    savefig('Phase3/Results/ActiveLearning.png')

def getResults(setting):
    results = []
    
    for trial in range(1, numTrials+1):
        results.append(list(load('Phase3/Results/'+setting+'/Results'+str(trial)+'.npy')))

    results = array(results).T
    stats = [round(mean(results[0]),3), round(std(results[0]),3), round(mean(results[3]),3), round(std(results[3]),3), round(mean(results[4]),3), round(std(results[4]),3), round(mean(results[5]),3), round(std(results[5]),3)]
    print(re.split('(?<=.)(?=[A-Z])', setting)[0]+' '+re.split('(?<=.)(?=[A-Z])', setting)[1]+' Statistics:', stats)
    save('Phase3/Results/'+setting+'/Stats', stats)

if argv[1][0] == 'P':
    # plotAccuraciesOverEpochs(argv[1])
    getResults(argv[1])
elif argv[1][0] == 'A':
    plotAccuraciesVsImagesLabeled(argv[1])
    getResults(argv[1])
elif argv[1] == 'ComparePassive':
    compareAccuraciesOverEpochs()
elif argv[1] == 'CompareActive':
    compareAccuraciesVsImagesLabeled()