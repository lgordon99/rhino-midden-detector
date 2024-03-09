# Rhino Midden Detector - IJCAI '23
<img src="https://github.com/lgordon99/rhino-midden-detector/blob/main/images/rhino-icon.png" width="33%" height="auto">

This is the GitHub repository for [Gordon et al.](https://www.ijcai.org/proceedings/2023/0663.pdf) To cite the paper, use the following BibTex.
```
@inproceedings{gordon_rhinos,
  title     = {Find Rhinos without Finding Rhinos: Active Learning with Multimodal Imagery of South African Rhino Habitats},
  author    = {Gordon, Lucia and Behari, Nikhil and Collier, Samuel and Bondi-Kelly, Elizabeth and Killian, Jackson A. and Ressijac, Catherine and Boucher, Peter and Davies, Andrew and Tambe, Milind},
  booktitle = {Proceedings of the Thirty-Second International Joint Conference on
               Artificial Intelligence, {IJCAI-23}},
  publisher = {International Joint Conferences on Artificial Intelligence Organization},
  editor    = {Edith Elkind},
  pages     = {5977--5985},
  year      = {2023},
  month     = {8},
  note      = {AI for Good},
  doi       = {10.24963/ijcai.2023/663},
  url       = {https://doi.org/10.24963/ijcai.2023/663},
}
```

## Modes
* passive (```p```)
* active (```a```)

## Modalities
* thermal (```t```)
* RGB (```r```)
* LiDAR (```l```)
* thermal-RGB fused (```tr-fused```)
* thermal-LiDAR fused (```tl-fused```)
* RGB-LiDAR fused (```rl-fused```)
* thermal-RGB-LiDAR fused (```trl-fused```)

## Queries
* random (```r```)
* uncertainty (```u```)
* positive certainty (```c```)
* disagree (```d```)
* multimodAL (```m```)

## File structure
code
* align-orthomosaics-and-middens.py: crops and aligns the orthomosaics and maps the midden locations onto them
* analyze-active-results.py: generates plots for accuracy and the fraction of middens found for the active learning methods
* analyze-passive-results.py: generates a plot for accuracy for the passive learning methods
* cnn.py: trains and tests one or more models
* cluster-middens.py: generates the midden map from a list of midden locations and identifies clusters using K-means
* fuse.py: generates the thermal-RGB, thermal-LiDAR, RGB-LiDAR, and thermal-RGB-LiDAR fusions
* merge-tiffs.py: if an orthomosaic is in the form of several tiffs, use this script to merge them into a single tiff
* midden-probability.py: calculates the probability of an image containing a midden given its maximum pixel value is no less than a threshold
* process-data.py: crops overlapping images from the orthomosaics and saves them along with their labels (midden/empty) and identifiers (integer)
* run-cnn.py: runs trials, including generating the train-test split, running the model, and saving the results
* view-images.py: view the image corresponding to a certain identifier

figures
* active-learning-accuracies.png: displays the accuracies achieved by various active learning methods on a held-out test set as the number of labeled images increases
* active-learning-fraction-middens-found.png: displays the fraction of middens discovered in the training set as the number of labeled images increases
* clustered-middens.png: midden map with clusters identified
* midden-probability.png: plots the probability of being a midden vs. the threshold on the maximum thermal pixel value
* passive-learning-accuracies.png: displays the accuracies achieved on a test set by models passively trained on different data modalities

## Instructions
1. Create a ```tiffs``` folder.
2. If your orthomosaics are in the form of several tiffs, use ```merge-tiffs.py``` to merge them into single tiffs.
3. Save the orthomosaic tiffs in the ```tiffs``` folder.
4. Create a ```data``` folder.
5. Save the midden coordinates as a CSV with "x" and "y" column headings in the ```data``` folder.
6. In the terminal, run the command
```bash
python align-orthomosaics-and-middens.py
```
Note you may need to change how the orthomosaics are aligned depending on their respective extents.\
7. In the terminal, run the command
```bash
python process-data.py
```
8. In the terminal, run the command
```bash
python fuse.py
```
9. In the terminal, run the command
```bash
python run-cnn.py TRIALS SAVE MODE MODALITY QUERY
```
where ```TRIALS``` is an integer, ```SAVE``` is a boolean, and ```MODE```, ```MODALITY```, and ```QUERY``` are selected from the lists above. ```QUERY``` can be left blank in passive mode (```MODE```=```p```).\
10. Once all trials have been run, use ```analyze-passive-results.py``` and ```analyze-active-results.py``` to generate the plots.
