lower_range=1
upper_range=30 # inclusive

for setting in 'ActiveUncertainty' 'ActiveCertainty' 'ActiveBinningThermal' 'ActiveBinningFused' 'ActiveBinningThermalRGB' 'ActiveDisagree' 'ActiveBinningThermalRGBLiDAR'; do
    sbatch --array=${lower_range}-${upper_range} Phase3/Bash/Job.sh $setting
done

# 'PassiveThermal' 'PassiveRGB' 'PassiveLiDAR' 'PassiveFused' 'ActiveUncertainty' 'ActiveCertainty' 'ActiveBinningThermal' 'ActiveBinningFused' 'ActiveBinningThermalRGB' 'ActiveDisagree' 'ActiveBinningThermalRGBLiDAR'