### Adaptive Clustering Evaluation

This repository contains the code to reproduce the results presented in the paper "Deep Clustering Evaluation: How to Validate Internal Validation Measures".

## Usage

### Set Up the Environment

Run the following command to install the required dependencies:

```bash
pip install -r requirements.txt
```

### ACE Evaluation

To replicate the results reported in the paper using the calculated measure values:

1. Download all the calculated internal measure values from the [Google Drive](https://drive.google.com/drive/folders/1FHehcJ8Qz7elY9IF3uu2JpMimH3G6l_D?usp=drive_link) and save them to a local folder.

2. Run the ACE evaluation script.

To run the default setting, simply execute the following command:

```bash
python ACE.py 
```
For more settings, run

```bash
python ACE.py --cl_method <method> --rank_method <method> --eps <value> --filter_alpha <value> --graph_alpha <value>
```

#### Arguments:
- `--cl_method`: Clustering method ('hdbscan', 'dbscan').
- `--rank_method`: Link analysis algorithm ('pr', 'hits').
- `--eps`: (Default: 0.05) Epsilon parameter for DBSCAN.
- `--filter_alpha`: (Default: 0.05) Family-wise error rate (FWER) for the Dip test.
- `--graph_alpha`: (Default: 0.05) FWER for creating the graph.

## Preliminary Steps for Evaluation

### Downloading Datasets

Download all the original datasets used to run and evaluate deep clustering algorithms from the [JULE repository](https://github.com/jwyang/JULE.torch).

### Get the External Measure Values

Ensure all downloaded datasets are stored in the `scripts/datasets` folder. Navigate to the `scripts` folder and run the following command to get the NMI and ACC values:

```bash
python get_truth.py --dataset <DATASET> --task <TASK>
```

Example for the JULE hyperparameter experiment on the COIL-20 dataset:

```bash
python get_truth.py --dataset COIL-20 --task jule
```

### Generate Internal Measure Values

All scripts to generate internal measure values for the evaluation are in the `scripts/embedded_metric` folder. Since some internal measure values can only be obtained through R packages, both R and Python scripts are used.

1. **Calculate Measure Values**:
   - `embedded_data.py`: Calculates measure values for the four internal measures reported in the main paper and prepares intermediate inputs for the R script.
   - `embedded.r`: Calculates the values for the remaining measures.
   - `collect_embedded_metric.py`: Post-processes the calculated values.

2. **Generate Shell Scripts for Slurms**:
   - `make.py`: Generates shell scripts for submission to Slurms, providing a reference for users on generating their submission scripts.

### Generate Raw Scores

Scripts for generating internal measure values used for the evaluation are in the `scripts/raw_metric` folder. Both R and Python scripts are utilized.

1. **Calculate Measure Values**:
   - `get_raw.py`: Calculates measure values for the four internal measures reported in the main paper and prepares intermediate inputs for the R script.
   - `getraw.r`: Calculates the values for the remaining measures.
   - `collect_raw.py`: Post-processes the calculated values.

2. **Generate Shell Scripts for Slurms**:
   - `make.py`: Generates shell scripts for submission to Slurms, providing a reference for users on generating their submission scripts.

### Dip Test

Scripts for performing the Dip test on embedding data obtained from JULE and DEPICT are in the `scripts/dip` folder.

### Selection of Checkpoint

Scripts for the selection of checkpoints and performing the Dip test on embedding data obtained from JULE and DEPICT are in the `scripts/DeepCluster` folder.
