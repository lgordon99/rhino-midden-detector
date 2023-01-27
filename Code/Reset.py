'''Reset by Lucia Gordon'''

from os import mkdir, path, remove
from shutil import rmtree

for setting in ['PassiveThermal', 'PassiveRGB', 'PassiveLiDAR', 'PassiveFused', 'ActiveUncertainty', 'ActiveCertainty', 'ActiveBinningThermal', 'ActiveBinningFused', 'ActiveBinningThermalRGB', 'ActiveDisagree', 'ActiveBinningThermalRGBLiDAR']:
    if path.exists('Phase3/Results/'+setting):
        rmtree('Phase3/Results/'+setting)

    mkdir('Phase3/Results/'+setting)

if path.exists('Phase3/Bash/Outputs'):
    rmtree('Phase3/Bash/Outputs')

if path.exists('Phase3/Bash/Errors'):
    rmtree('Phase3/Bash/Errors')

mkdir('Phase3/Bash/Outputs')
mkdir('Phase3/Bash/Errors')