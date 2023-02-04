<h1 align='center'>QSAR Bioconcetration Project</h1>

Deep learning model predicting class of bioocontentration of a chemical compund.
The model predicts tendency of a given compund to be stored in living organisms, assigning to one of bioconcentration classes:
- Class 1: Compund mainly stored within lipid tissue
- Class 2: Compound has additional storage sites (e.g. plasma proteins)
- Class 3: Compound is metabolised or eliminated, bioconcentration reduced

This project was based on a 2016 paper by F. Grisoni et al. "Investigating the mechanisms of bioconcentration through QSAR
classification trees".[^1]

## Background
**QSAR (quantitative structure–activity relationship) modeling** is an _in silico_ approach of examination of biological properties of compunds. QSAR models base on theoretical parameters called <b>molecular descriptors</b> that can be calculated directly from chemical structures of given molecules. Therefore this method may be used to evaluate properties of a chemical compund without the need of synthesising it in a lab or collecting from nature or even predict properties of theoretical chemical structures that have never existed before. All these features make QSAR models less costly and time-consuming alternatives to experiments on cell lines or animals in the fields like toxicity assessments of chemicals or drug discovery.

**Bioconcentration** is the intake and retention of a substance in an organism entirely by respiration from water in aquatic ecosystems or from air in terrestrial ones. This property may be used for assessment of environmental safety of potentially dangerous chemicals.

## Dataset
The dataset consists of 779 records, each representing one chemical compund. Every record is identified with CAS registry number and descried with chemical structure in SMILES format and 9 molecular descriptors. The compounds were also labeled with class number (1-3), logBCF (dependent value for regression problem) and assignment to test or train subset.
The dataset was created by F. Grisoni et al. Compounds were assigned to bioconcentration classes by the authors on the basis of wet weight BCF data and the TGD (Technical Guidance Document) model.

### Molecular descriptors
The molecular descriptors were calculated directly from chemical structures in SMILES format using Dragon software. The table below shows names and descriptions of used molecular descriptors.[^2]

|    Name   |                            Description                             |
| --------- | ------------------------------------------------------------------ |
|nHM        | number of heavy atoms Constitutional indices                       |
|piPC09     | molecular multiple path count of order 9                           |
|PCD        | difference between multiple path count and path count              |
|X2Av       | average valence connectivity index of order 2                      |
|MLOGP      | Moriguchi octanol-water partition coeff. (logP)                    |
|ON1V       | overall modified Zagreb index of order 1 by valence vertex degrees |
|N-072      | RCO-N< / >N-X=X                                                    |
|B02[C-N]   | Presence/absence of C - N at topological distance 2                |
|F04[C-O]   | Frequency of C - O at topological distance 4                       |

## Files description

### Notebook
#### **data_analysis.ipynb**
The notebook showing basic dataset information with analysis of class proportion, outliers presence and correlation between features.

### Utility files
#### **preprocess.py**
Script containing Dataset class definition. Objects of Dataset type have data loaded from a file as _pandas.DataFrame_ and set of methods for data health check and  preprocessing:

- ```check_descriptors_completeness```

  Checks if all molecular descriptors needed for model training and testing are present in dataset (as defined in self.descriptors attribute).

- ```split_data```

  Method splitting dataset either to:
  - features subset (self.X) and class labels or dependent variables subset (self.y)
  - training, validation and test subsets.
  
  User needs to specify class labels column (_class_column_ parameter) and dependent variables (_reg_column_ parameter) and whether the dataset is being splitted for classification or linear regression model.
  
- ```remove_outliers```

  ***Important***: if user wants to perform both removing outliers and scaling, this method should be used first.

  Method searches outliers in columns with floating point values and removes records containing them. By default the method recognizes values higher or lower by 3 standard deviations than the mean value as outliers.

- ```scale```

  Perform standardization or normalization of columns with floating point values as _sklearn.preprocessing.StandardScaler_ object or _sklearn.preprocessing.MinMaxScaler_ object respectively.

- ```resample```

  Fixing training on imbalanced classes with SMOTE resampling. SMOTE (Synthetic Minority Over-sampling Technique) consists in generating synthetic data to over-sample minority class using k nearest neighbors algorithm.[^3]

- ```adjust_shape```

  Changing shape of class labels column to format required in sequential model.
 
#### **tools.py**
This script is a place for functions that don't take part in ETL process

#### config.yaml
Configuration file for ETL and model training parameters


### Executable scripts
#### etl.py
The script contains a function which loads raw data, splits, processes and exports it

#### train.py
Defining sequential model architecture, compiling, training and exporting created model.











### References
[^1]: Grisoni F., Consonni V., Vighi M., Villa S., Todeschini R. Investigating the mechanisms of bioconcentration through QSAR classification trees. Environ. Int. 2016;88:198–205. [DOI](https://doi.org/10.1016/j.envint.2015.12.024) - [PubMed](https://pubmed.ncbi.nlm.nih.gov/26760717/)
[^2]: [List of molecular descriptors calculated by Dragon software](http://talete.mi.it/products/dragon_molecular_descriptor_list.pdf)
[^3]: N. V. Chawla, K. W. Bowyer, L. O.Hall, W. P. Kegelmeyer, “SMOTE: synthetic minority over-sampling technique,” Journal of artificial intelligence research, 321-357, 2002 [DOI](https://doi.org/10.1613/jair.953)
