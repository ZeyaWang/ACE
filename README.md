# Adaptive Clustering Evaluation
Adaptive Clustering Evaluation (ACE) is an internal evaluation method designed to assess clustering results produced by deep clustering algorithms. It introduces a weighted scoring mechanism that combines internal scores, computed across different embedding spaces derived from the deep clustering process. This repository provides the code necessary to reproduce the results presented in the corresponding paper.["Deep Clustering Evaluation: How to Validate Internal Validation Measures"](https://arxiv.org/abs/2403.14830).

## Usage
The code for simulation and real data analysis can be found in the `./simulation/*` and `./real_data/*` directories, respectively.


### Set Up the Environment

Run the following command to install the required dependencies:

```bash
pip install -r requirements.txt
```


### Simulation
To run the simulation and reproduce the results presented in the paper, navigate to the `./simulation/*` directory and follow these steps:  

#### **1. Generate Simulations for Different Settings**  
Each simulation can be generated using the following command:  
```bash
python sim.py --option <simulation_setting> --cluster_std <sample_standard_deviation> --seed <random_seed> 
```  
##### **Arguments:**  
- `--option`: Specifies the simulation setting (`dense` or `sparse`).  
- `--cluster_std`: Defines the sample standard deviation.  
- `--seed`: Sets the random seed for reproducibility.  

To automate the generation of all simulations, run the script `make_sim.py`. This will create shell scripts for submitting jobs to a SLURM cluster, ensuring that all simulated datasets are generated efficiently.  

#### **2. Compute Internal Scores and Dip Test Results**  
For each simulated dataset, internal scores and dip test results must be computed across all embedding spaces.  

- **Internal Scores:**  The script `calculate_metrics.py` computes internal scores for a single simulated dataset using one of the metrics reported in the main text. Run `make_metrics.py` to generate shell scripts for submitting jobs to a SLURM cluster, automating the computation of internal scores across all datasets and metrics.

- **Dip Test Results:**  The script `clusterable.R` generates the dip test results for a single simulated dataset.  Run `make_dip.py` to generate shell scripts for submitting jobs to a SLURM cluster, automating dip test computations across all datasets.  

#### **3. Compute Final Weighted Scores and Generate Plots**  
- Firstly run `ACE.py` to compute and save the final weighted scores produced by ACE.  
- Then run `boxplot.py` to generate the main figure for simulations from the saved results, as presented in the paper.


### Evaluation for Real Data Analysis

To replicate the results reported in the paper using the calculated measure values:

1. Download all the calculated internal measure values from the [Google Drive](https://drive.google.com/drive/folders/1FHehcJ8Qz7elY9IF3uu2JpMimH3G6l_D?usp=drive_link) and save them to a local folder.

2. Run the ACE evaluation script.

To run the default setting, simply execute the following command:

```bash
python ACE.py 
```
For more settings, run

```bash
python ACE.py --cl_method <grouping_method> --rank_method <ranking_method> --eps <dbscan_thresh> --filter_alpha <dip_fwer> --graph_alpha <graph_fwer>
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
