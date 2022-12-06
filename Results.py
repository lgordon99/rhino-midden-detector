from matplotlib.pyplot import figure, imshow, xlabel, xticks, ylabel, scatter, plot, show, legend, savefig
from numpy import load, mean

class Results:
    def __init__(self):
        #self.results = load(resultsPath)

        self.plotResults()
        #print(self.results)


    def plotResults(self):
        '''F1-Score'''
        # figure('F1-Score Results', dpi=150)
        # avg = mean([0,0.626,0.651,0.767,0.645])
        # scatter(range(1,6),[0,0.626,0.651,0.767,0.645], c='b')
        # plot(range(1,6),[0,0.626,0.651,0.767,0.645], c='b')
        # plot(range(1,6),[avg,avg,avg,avg,avg], c='g', label='mean')
        # ylabel('F1-Score')
        
        '''Recall'''
        figure('Recall Results', dpi=150)
        avg = mean([0,0.468,1,0.77,0.849])
        scatter(range(1,6),[0,0.468,1,0.77,0.849], c='b')
        plot(range(1,6),[0,0.468,1,0.77,0.849], c='b')
        plot(range(1,6),[avg,avg,avg,avg,avg], c='g', label='mean')
        ylabel('Recall')
        
        xticks(range(1,6))
        xlabel('Trial')
        legend()
        #savefig('Results/F1-Score.png')
        show()

Results()