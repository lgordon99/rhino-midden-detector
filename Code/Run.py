'''Run by Lucia Gordon'''

from numpy import load, save
from sys import argv
from TrainTestSplit import TrainTestSplit
from CNN import CNN

def runCNN(setting, trial):
    print('Setting =', setting)

    if setting[0] == 'P':
        balance = True
    elif setting[0] == 'A':
        balance = False

    print('Trial ' + str(trial))
    indices = TrainTestSplit(setting, str(trial), balance).indices
    model = CNN(setting, indices)
    save('Phase3/Results/'+setting+'/Results'+str(trial), model.results)
    
    if setting[0] == 'P':
        save('Phase3/Results/'+setting+'/AccuracyOverEpochs'+str(trial), model.accuracies)
    elif setting[0] == 'A':
        save('Phase3/Results/'+setting+'/AccuracyVsImagesLabeled'+str(trial), model.accuracies)

runCNN(setting=argv[1], trial=int(argv[2]))