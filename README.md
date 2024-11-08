<h1 align='center'>QSAR Bioconcetration Project</h1>

Deep learning model predicting class of bioocontentration of a chemical compund.
The model predicts tendency of a given compund to be stored in living organisms, assigning to one of bioconcentration classes:
- Class 1: Compund mainly stored within lipid tissue
- Class 2: Compound has additional storage sites (e.g. plasma proteins)
- Class 3: Compound is metabolised or eliminated, bioconcentration reduced

This project was based on a 2016 paper by F. Grisoni et al. "Investigating the mechanisms of bioconcentration through QSAR
classification trees".[^1]

## Background
**QSAR (quantitative structure–activity relationship) modeling** is an _in silico_ approach of examination of biological properties of chemical compunds. QSAR models base on theoretical parameters called <b>molecular descriptors</b> that can be calculated directly from chemical structures of given molecules. Therefore this approach may be used to evaluate properties of a chemical compund without the need of synthesising it in a lab or collecting from nature or even predict properties of theoretical chemical structures that have never existed before. All these features make QSAR models less costly and time-consuming alternatives to experiments on cell lines or animals in the fields like toxicity assessments of chemicals or drug discovery.

**Bioconcentration** is the intake and retention of a substance in an organism entirely by respiration from water in aquatic ecosystems or from air in terrestrial ones. This property may be used for assessment of environmental safety of potentially dangerous chemicals.

## Dataset
The dataset was created by F. Grisoni et al. and attached to the 2016 paper[^1] as supplementary data.

The dataset consists of 779 records, each representing one chemical compund. Every record is identified with CAS registry number and described with chemical structure in SMILES format, 9 molecular descriptors and logBCF (dependent value for regression problem). The compounds were assigned to bioconcentration classes by the authors on the basis of wet weight BCF data and the TGD (Technical Guidance Document) model and labeled with one of class numbers (1-3).

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

### References
[^1]: Grisoni F., Consonni V., Vighi M., Villa S., Todeschini R. Investigating the mechanisms of bioconcentration through QSAR classification trees. Environ. Int. 2016;88:198–205. [DOI](https://doi.org/10.1016/j.envint.2015.12.024) - [PubMed](https://pubmed.ncbi.nlm.nih.gov/26760717/)
[^2]: [List of molecular descriptors calculated by Dragon software](http://talete.mi.it/products/dragon_molecular_descriptor_list.pdf)
[^3]: N. V. Chawla, K. W. Bowyer, L. O.Hall, W. P. Kegelmeyer, “SMOTE: synthetic minority over-sampling technique,” Journal of artificial intelligence research, 321-357, 2002 [DOI](https://doi.org/10.1613/jair.953)
