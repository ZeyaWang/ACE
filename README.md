# Adaptive Clustering Evaluation

The code is provided for reproducing the results in the paper "Deep Clustering Evaluation: How to Validate Internal Validation Measures".

# Usage

## ACE Evaluation

To replicate the results reported in the main paper given the calculated measure values

* Download all the calculated internal measure values from the [Google drive](https://drive.google.com/drive/folders/1FHehcJ8Qz7elY9IF3uu2JpMimH3G6l_D?usp=drive_link) to the local folder

* Run `python ACE.py`. The arguments can be specified to change the setting of running our ACE.
  - --cl_method: ('hdbscan','dbscan'); different choice of grouping method
  - --rank_method', ('pr', 'hits'); different choice of link analysis algorithms
  - --eps, default=0.05, eps required for dbscan
  - --filter_alpha', default=0.05, FWER for dip test
  - --graph_alpha', default=0.05, FWER for creating graph


## Preliminary Steps for Evaluation

### Datasets downloading

All the original datasets used for run and evaluate deep clustering algorithms can be downloaded from the Github repository of [JULE](https://github.com/jwyang/JULE.torch).

### Get the external measure values

Given all the downloaded datasets are stored in the folder `scripts/datasets`, please go the folder `scripts` and run `python get_truth.py --dataset $DATASET --task $TASK` to get the NMI and ACC values. For example, for the results obtained from JULE hyperparameter experiment on the dataset COIL-20, run

`python get_truth.py --dataset COIL-20 --task jule`

### Generate the internal measure values for all the pairs of partitioning results and embedded data

All the scripts to generate all the internal measure values used for the evaluation are in the folder script/embedded_metric. Consider some internal measure values can only be obtained through R package, we use both R and Python scripts to generate all these measure values. 

* The script `embedded_data.py` is used to calculate the measure values for the four internal measures reported in the main paper and prepare the intermediate inputs to the R script for more efficient calculation.

* The script 'embedded.r' is used to calculate the values for the remaining measures.

* The script 'collect_embedded_metric.py' is used to postprocess the calculated values.

* The script 'make.py' is a script to generate shell scripts submitted to Slurms, which can provide a reference to users about how to generate their submission scripts.


### Generate raw scores.
All the scripts to generate all the internal measure values used for the evaluation are in the folder script/raw_metric. Consider some internal measure values can only be obtained through R package, we use both R and Python scripts to generate all these measure values. 

* The script `get_raw.py` is used to calculate the measure values for the four internal measures reported in the main paper and prepare the intermediate inputs to the R script for more efficient calculation.

* The script 'getraw.r' is used to calculate the values for the remaining measures.

* The script 'collect_raw.py' is used to postprocess the calculated values.

* The script 'make.py' is a script to generate shell scripts submitted to Slurms, which can provide a reference to users about how to generate their submission scripts.

### Dip test
All the scripts to perform Dip test for the embedding data obtained from JULE and DEPICT are stored in the folder `scripts/dip`.


### Selection of checkpoint
For the experiments for the selection of checkpoint, all the scripts to perform Dip test for the embedding data obtained from JULE and DEPICT are stored in the folder `scripts/DeepCluster`.




