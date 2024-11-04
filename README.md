# mbo-fs
The code repository for the paper "The performance of Migrating Birds Optimization as a feature selection tool"

The datasets are taken from the [UC Irvine Machine Learning Repository](https://archive.ics.uci.edu/), they are distributed under 
the [Creative Commons Attribution 4.0 International (CC BY 4.0) license](https://creativecommons.org/licenses/by/4.0/legalcode). 
They are:

 - Mesothelioma's disease dataset ([link](https://doi.org/10.24432/C5903T)).
 - Glioma Grading Clinical and Mutation Features dataset ([link](https://doi.org/10.24432/C5R62J)).
 - DARWIN dataset ([link](https://doi.org/10.24432/C55D0K)).
 - Breast Cancer Wisconsin (Diagnostic) dataset ([link](https://doi.org/10.24432/C5DW2B)).

Their abbreviations in the script names are dicle, glioma, darwin and wbc, respectively. The files `[datasetname].py` are for the 
MBO and HHO feature selections, while `[datasetname]_pca.py` are for the PCA-related methods and `[datasetname]_other.py` are for 
the remaining feature selection algorithms.

The file `mdset.xlsx` is for dicle, while the `darwin` and `glioma` subdirectories include the files of the corresponding 
datasets.

The Python code used requires NumPy and scikit-learn.
