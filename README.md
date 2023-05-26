README

Settings
Modes: passive ("p") and active ("a")
Modalities: thermal ("t"), RGB ("r"), LiDAR ("l"), thermal-RGB fused ("tr-fused"), thermal-LiDAR fused ("tl-fused"), RGB-LiDAR fused ("rl fused"), and thermal-RGB-LiDAR fused ("trl-fused")
Queries: random ("r"), uncertainty ("u"), positive certainty ("c"), disagree ("d"), and multimodAL ("m")

File structure
1. code
1a. align-orthomosaics-and-middens.py: crops and aligns the orthomosaics and maps the midden locations onto them
1b. analyze-active-results.py: generates plots for accuracy and the fraction of middens found for the active learning methods
1c. analyze-passive-results.py: generates a plot for accuracy for the passive learning methods
1d. cnn.py: trains and tests one or more models
1e. cluster-middens.py: generates the midden map from a list of midden locations and identifies clusters using K-means
1f. fuse.py: generates the thermal-RGB, thermal-LiDAR, RGB-LiDAR, and thermal-RGB-LiDAR fusions
1g. merge-tiffs.py: if an orthomosaic is in the form of several tiffs, use this script to merge them into a single tiff
1h. midden-probability.py: calculates the probability of an image containing a midden given its maximum pixel value is no less than a threshold
1i. process-data.py: crops overlapping images from the orthomosaics and saves them along with their labels (midden/empty) and identifiers (integer)
1j. run-cnn.py: runs trials, including generating the train-test split, running the model, and saving the results
1k. view-images.py: view the image corresponding to a certain identifier

2. figures
2a. active-learning-accuracies.png: displays the accuracies achieved by various active learning methods on a held-out test set as the number of labeled images increases
2b. active-learning-fraction-middens-found.png: displays the fraction of middens discovered in the training set as the number of labeled images increases
2c. clustered-middens.png: midden map with clusters identified
2d. midden-probability.png: plots the probability of being a midden vs. the threshold on the maximum thermal pixel value
2e. passive-learning-accuracies.png: displays the accuracies achieved on a test set by models passively trained on different data modalities

Instructions
1. Create a "tiffs" folder.
2. If your orthomosaics are in the form of several tiffs, use "merge-tiffs.py" to merge them into single tiffs.
3. Save the orthomosaic tiffs in the "tiffs" folder.
4. Create a "data" folder.
5. Save the midden coordinates as a CSV with "x" and "y" column headings in the "data" folder.
6. In the terminal, run the command "python align-orthomosaics-and-middens.py". Note you may need to change how the orthomosaics are aligned depending on their respective extents.
7. In the terminal, run the command "python process-data.py".
8. In the terminal, run the command "python fuse.py".
9. In the terminal, run the command "python run-cnn.py TRIALS SAVE MODE MODALITY QUERY", where "TRIALS" is an integer, "SAVE" is a boolean, and "MODE," "MODALITY," and "QUERY" are selected from the lists above. "QUERY" can be left blank in passive mode ("MODE"="p").
10. Once all trials have been run, use "analyze-passive-results.py" and "analyze-active-results.py" to generate the plots.