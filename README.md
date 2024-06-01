# Adaptive Clustering Evaluation

The code is provided for reproducing the results in the paper "Deep Clustering Evaluation: How to Validate Internal Validation Measures".

# Usage
To replicate the results reported in the paper

* Download all the calculated internal measure values from the [Google drive](https://drive.google.com/drive/folders/1FHehcJ8Qz7elY9IF3uu2JpMimH3G6l_D?usp=drive_link) to the local folder

* Run `python ACE.py`. The arguments can be specified to change the setting of running our ACE.
  -cl_method: ('hdbscan','dbscan'); different choice of grouping method
  -rank_method', ('pr', 'hits'); different choice of link analysis algorithms
  -eps, default=0.05, eps required for dbscan
  -filter_alpha', default=0.05, FWER for dip test)
  -graph_alpha', default=0.05, FWER for creating graph)