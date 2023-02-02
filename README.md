<h1 align='center'>QSAR Bioconcetration Project</h1>

Deep learning model predicting class of bioocontentration of a chemical compund.
The model predicts tendency of a given compund to be stored in living organisms, assigning to one of bioconcentration classes:
- Class 1: Compund mainly stored within lipid tissue
- Class 2: Compound has additional storage sites (e.g. plasma proteins)
- Class 3: Compound is metabolised or eliminated, bioconcentration reduced

This project was based on 2016 paper by F. Grisoni et al. "Investigating the mechanisms of bioconcentration through QSAR
classification trees"

<h2>Background</h2>
<p><b>QSAR (quantitative structure–activity relationship) modeling</b> is an <i>in silico</i> approach of examination of biological properties of compunds. QSAR models base on theoretical parameters called <b>molecular descriptors</b> that can be calculated directly from chemical structures of given molecules. Therefore this method may be used to evaluate properties of a chemical compund without the need of synthesising it in a lab or collecting from nature or even predict properties of theoretical chemical structures that have never existed before. All these features make QSAR models less costly and time-consuming alternatives to experiments on cell lines or animals in the fields like toxicity assessments of chemicals or drug discovery.</p>

<p><b>Bioconcentration</b> is the intake and retention of a substance in an organism entirely by respiration from water in aquatic ecosystems or from air in terrestrial ones. This property may be used for assessment of environmental safety of potentially dangerous chemicals.</p>

<h2>Dataset</h2>
The dataset consists of 779 records, each representing one chemical compund. Every record is described with CAS registry number, chemical structure in SMILES format and 9 molecular descriptors. The compounds were also labeled with class number (1-3), logBCF (dependent value for regression problem) and assignment to test or train subset.
The dataset was created by F. Grisoni et al. The molecular descriptors were calculated directly from chemical structure in SMILES format using Dragon software. The compounds were assigned to bioconcentration classes by the authors on the basis of wet weight BCF data and the TGD (Technical Guidance Document) model.

<h2>Molecular descriptors</h2>

