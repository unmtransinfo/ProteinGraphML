# ProteinGraphML

This software is designed to predict to predict disease-to-protein (protein-coding
gene) associations, from a biomedical knowledge graph, via machine learning (ML).
This codebase abstracts the ML from the domain knowledge and data sources, to
allow reuse for other applications. The input PostgreSQL relational database is
converted to a knowledge graph, then converted to feature vectors by metapath
matching, based on an input disease, defining a training set of proteins. Then
XGBoost is used to generate and optimize a predictive model.

## Table of Contents  

* [Dependencies](#Dependencies)
* [How to Run Workflow](#Howto)
   * [Build KG](#HowtoBuildKG)
   * [Generate Metapath Features](#HowtoMetapathFeatures)
   * [Generate Static Features](#HowtoStaticFeatures)
   * [Prepare Training and Test Sets](#HowtoPrep)  _(For custom labeled training set.)_
   * [Train ML Model](#HowtoTrainML)
   * [Test Trained Model](#HowtoPredictML)
   * [Visualization](#HowtoVis) _(Optional.)_

## <a name="Dependencies"/>Dependencies

* R 3.5+
* R packages: `data.table`, `Matrix`, `RPostgreSQL`
* Python 3.4+
* Python packages: `xgboost`, `scikit-learn`, `networkx`, `pandas`, `pony`, `matplotlib`, `xlrd`, `XlsxWriter`
* PostgreSQL database `metap`.
   * Edit `DBcreds.yaml` with valid db credentials. Needed throughout workflow.

## <a name="Howto"/>How to run the Workflow:

The command-line programs of ProteinGraphML must be executed in the following order.
However, the __BuildKG__ and __StaticFeatures__ are one-time steps, re-useable for
multiple ML models. Re-run only required if database updated.

### <a name="HowtoBuildKG"/>Build KG

`BuildKG_OlegDb.py`, from the relational db, generates a knowledge graph,
a `ProteinDiseaseAssociationGraph`, saved as a pickled `networkX` graph. 
Via the adaptor and [`pony`](https://docs.ponyorm.org)
object-relational model (ORM), nodes and edges are queried from the db to comprise the
graph.

Command line parameters:

* `operation` (positional parameter):
   * `build` :  build KG and write to output file.
   * `test` : build KG, write log, but not output file.
* `--ofile` : Pickled KG file (default: ProteinDisease_GRAPH.pkl).

Example commands:

```
BuildKG_OlegDb.py -h
BuildKG_OlegDb.py test
BuildKG_OlegDb.py build
```

### <a name="HowtoPrep"/>Prepare Training and Test Sets _(For custom labeled training set.)_

`PrepTrainingAndTestSets.py` generates two files:

1.  A `pickle`ed Python dictionary that
contains protein_ids for both class 'True' and 'False'. This training set file is needed
for running ML for a disease defined by a custom labeled training set,
rather than a Mammalian Phenotype (MP) term ID. The custome labeled training set may
reference proteins via `protein_id`s or gene symbols; if gene symbols, this code fetches
the corresponding `protein_id` for each symbol from the database. The prepared,
picked training set uses `protein_id`s.
1. A `pickle`ed test set of protein_ids, unlabeled, defining the predictions of
interest, for testing the trained model.

Command line parameters:

* `--file` : File that contains protein_ids/symbols and labels for a given disease, with extension (.txt|.xlsx|.rds).
* `--dir` : directory where data files are found (default: DataForML).
* `--symbol_or_pid` : "symbol" or "pid" (default: symbol).
* `--use_default_negatives` : Use default negatives, ~3500 genes with known associations but not with query disease. If false, input training set must include negatives.

If the file is a spreadsheet, the header should have "Protein_id Label" or "Symbol Label".
If the file is a text file, the Protein_id/symbol and
Label should be comma-separated. There should not be any header in the text file. If the
file is an RDS file, the parameter 'symbol'  can be omitted. Use one of the following,
to run this program.

Example commands:

```
PrepTrainingAndTestSets.py -h
PrepTrainingAndTestSets.py --file diabetes_pid.txt --symbol_or_pid 'pid'
PrepTrainingAndTestSets.py --file autophagy_test20191003.xlsx
PrepTrainingAndTestSets.py --file 125853.rds
PrepTrainingAndTestSets.py --file diabetes.xlsx --use_default_negatives
```

### <a name="HowtoMetapathFeatures"/>Metapath Features

To generate metapath features from the KG, use `GenerateFeatures.py`. From the KG
and hard coded metapath patterns, plus the positively labeled proteins in the
training set, feature vectors are generated for all training cases and optionally
test cases. Normally, any human proteins not in the labeled training set 
will be in the test set.  Metapath-based features
must be generated for each model (unlike static features), since how metapath 
semantic patterns match the KG depends on the query disease.

```
./GenerateFeatures.py --trainingfile DataForML/autophagy_test20191003.pkl --test DataForML/autophagy_test20191003_test.pkl --outputdir results/autophagy/
```

### <a name="HowtoStaticFeatures"/>Static Features

To generate static features (not metapath-based), use R script
`ProteinGraphML/MLTools/StaticFeatures/staticFiles.R` to generate CSV files, for ccle,
gtex, lincs and hpa, for use by `TrainModelML.py`. Static features are _not_ dependent on
trainging set labels, only the database, so the same CSV files can be reused for
all models, and only needs to be re-run if the database changes.

```
cd ProteinGraphML/MLTools/StaticFeatures
./staticFiles.R
```

### <a name="HowtoTrainML"/>Train ML Model

`TrainModelML.py`, from the training set feature vectors, or a training set
implicated by specified disease (Mammalian Phenotype ID), 
executes the specified ML procedure, training a predictive model, then saved to a
reusable file (.model).  The procedure `XGBCrossVal` uses
XGBoost, trains a model with cross-validation and grid-search parameter optimization,
generates a list of important features used by the classification model.

Command line parameters:

* `procedure` (positional parameter):
   * `XGBCrossValPred` :  5-fold cross-validation, one iteration.
   * `XGBCrossVal` : 5-fold cross-validation, multiple iterations.
* `--disease` : Use with Mammalian Phenotype ID, e.g. MP_0000180.
* `--file` : Training set file, produced by `PrepTrainingAndTestSets.py`.
* `--resultdir` : Name of the directory where results will be stored (e.g. MP_0000180).
* `--kgfile` : KG file, produced by `BuildKG_OlegDb.py` (default: ProteinDisease_GRAPH.pkl).

Example commands:

```
TrainModelML.py -h
TrainModelML.py XGBCrossVal --file 144700.pkl --resultdir 144700
TrainModelML.py XGBCrossValPred --file 144700.pkl --resultdir 144700
TrainModelML.py XGBCrossVal --disease MP_0000180 --resultdir MP_0000180
TrainModelML.py XGBCrossValPred --disease MP_0000180 --resultdir MP_0000180
```

Results will be saved in the specified --resultsdir. See logs for specific
subdirectories and output files, including:

* Saved XGBoost model (.model).
* Feature importance lists (.tsv, .xlsx).

### <a name="HowtoPredictML"/>Test Trained ML Model

`PredictML.py`, Using the model trained on the training set and KG, predicts the probability of True class for
proteins. The procedure `XGBPredict` uses the saved XGBoost model, and generates results for predictions on all proteins in the test set. 

Command line parameters:

* `procedure` (positional parameter):
   * `XGBPredict` :  load the saved model
* `--model` : Full path to the trained model (results/autophagy_test20191003/XGBCrossVal.model).
* `--file` : Test set file, produced by `PrepTrainingAndTestSets.py`.
* `--kgfile` : KG file, produced by `BuildKG_OlegDb.py` (default: ProteinDisease_GRAPH.pkl).

Example commands:

```
PredictML.py -h
PredictML.py XGBPredict --file autophagy_test20191003_test.pkl --model results/autophagy_test20191003/XGBCrossVal.model
```

Results will be saved in the specified --resultsdir. See logs for specific
subdirectories and output files, including:

* Predictions with probabilities for all proteins (.tsv, .xlsx).

### <a name="HowtoVis"/>Visualize _(optional)_

`MakeVis.py` generates HTML/JS files, with feature importance, for visualization
via web browser.  `ProteinGraphML/Analysis/` contains code for graph and feature
visualization. Visualize has code for creating HTML/JS graphs, and featureLabel has
code for taking a dictionary of feature importance, and giving it human readable labels.

Command-line parameters:

* `--disease` : Disease name.
* `--featurefile` : full path to the pickled features file produced by `TrainModelML.py`, e.g. results/104300/featImportance_XGBCrossVal.pkl.
* `--num` : number of top important features selected.
* `--kgfile` : Pickled KG file, produced by `BuildKG_OlegDb.py` (default: ProteinDisease_GRAPH.pkl).

Example command:

```
MakeVis.py --disease 104300 --featurefile results/104300/featImportance_XGBCrossVal.pkl --num 2
```

## <a name="Notes"/>Notes

* The code currently assumes that all nodes are unique, that proteins are integer IDs, and the only ints in the graph. 
* New data sources can be supported by adding new Adapter class in `ProteinGraphML/DataAdapter/`.
* New ML procedures may be added to `ProteinGraphML/MLTools/Procedures/`.

Workflow diagram:

> <img src="doc/MetapathDiagram.png" height="400">
