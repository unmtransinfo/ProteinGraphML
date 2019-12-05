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

* `OPERATION` (positional parameter):
   * `build` :  build KG and write to output file.
   * `test` : build KG, write log, but not output file.
* `--ofile` : Pickled KG file (default: ProteinDisease_GRAPH.pkl).
* `--logfile` : KG log file (default: ProteinDisease_GRAPH.log).

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

* `--i` : Input file that contains protein_ids/symbols and labels for a given disease, with extension (csv|txt|xlsx|rds).
* `--symbol_or_pid` : "symbol" or "pid" (default: symbol).
* `--use_default_negatives` : Use default negatives, ~3500 genes with known associations but not with query disease. If false, input training set must include negatives.

If the file is a spreadsheet, the header should have "Protein_id Label" or "Symbol Label".
If the file is a text file, the Protein_id/symbol and
Label should be comma-separated. There should not be any header in the text file. 

Example commands:

```
PrepTrainingAndTestSets.py -h
PrepTrainingAndTestSets.py --i data/diabetes_pid.txt --symbol_or_pid 'pid'
PrepTrainingAndTestSets.py --i data/autophagy_test20191003.xlsx
PrepTrainingAndTestSets.py --i data/diabetes.xlsx --use_default_negatives
PrepTrainingAndTestSets.py --i data/Asthma.rds
```

### <a name="HowtoMetapathFeatures"/>Metapath Features

To generate metapath features from the KG, use `GenTrainingAndTestFeatures.py`. From the KG
and hard coded metapath patterns, plus the positively labeled proteins in the
training set, feature vectors are generated for all training cases and optionally
predict cases. Normally, any human proteins not in the labeled training set 
will be in the predict set.  Metapath-based features
must be generated for each model (unlike static features), since how metapath 
semantic patterns match the KG depends on the query disease.

Command line parameters:

* `--disease` : Mammalian Phenotype ID, e.g. MP_0000180
*  `--trainingfile` : pickled training set, e.g. "diabetes.pkl"
*  `--predictfile` : pickled predict set, e.g. "diabetes_test.pkl"
*  `--outputdir` : directory where train and test data with features will be saved, e.g. "diabetes_no_lincs"
*  `--kgfile` : input pickled KG (default: "ProteinDisease_GRAPH.pkl")
*  `--static_data` : (default: "gtex,lincs,ccle,hpa")

Example commands:

```
GenTrainingAndTestFeatures.py -h
GenTrainingAndTestFeatures.py --trainingfile data/autophagy_test20191003.pkl --predictfile data/autophagy_test20191003_test.pkl --outputdir results/autophagy/
GenTrainingAndTestFeatures.py --disease MP_0000180 --outputdir results/MP_0000180
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

* `PROCEDURE` (positional parameter):
   * `XGBGridSearch` :  Grid search for optimal XGBoost parameters.
   * `XGBCrossValPred` :  5-fold cross-validation, one iteration.
   * `XGBKfoldsRunPred` : 5-fold cross-validation, multiple iterations.
* `--crossval_folds` : number of cross-validation folds
* `--xgboost_param_file` : XGBoost configuration parameter file (e.g. XGBparams.txt)
* `--trainingfile` : Training set file, produced by `PrepTrainingAndTestSets.py`.
* `--resultdir` : directory for output results
* `--kgfile` : KG file, as produced by `BuildKG_OlegDb.py` (default: ProteinDisease_GRAPH.pkl).

Example commands:

```
TrainModelML.py -h
TrainModelML.py XGBCrossValPred --trainingfile results/144700.pkl --resultdir results/144700
```

Results will be saved in the specified --resultsdir. See logs for specific
subdirectories and output files, including:

* Saved XGBoost model (.model).
* Feature importance lists (.tsv, .xlsx).

### <a name="HowtoPredictML"/>Test Trained ML Model

`PredictML.py`, Using the model trained on the training set and KG, predicts the probability of True class for
proteins. The procedure `XGBPredict` uses the saved XGBoost model, and generates results for predictions on all proteins in the test set. 

Command line parameters:

* `PROCEDURE` (positional parameter):
   * `XGBPredict` :  load the saved model
* `--modelfile` : trained model (e.g. results/autophagy_test20191003/XGBCrossVal.model).
* `--predictfile` : predict data file, produced by `PrepTrainingAndTestSets.py` (e.g.  "diabetesPredictData.pkl")
* `--resultdir` : directory for output results

Example commands:

```
PredictML.py -h
PredictML.py XGBPredict --predictfile autophagy_test20191003_predict.pkl --model results/autophagy_test20191003/XGBCrossVal.model --resultdir results/autophagy
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

## <a name="CompareRandPythonPrediction"/>Compare R and Python Predictions/Features
### <a name="GenerateFeaturesFiles"/>Generate features files for training and prediction data using RDS file
`CreateFeatureFilesFromRDS.py` takes RDS file as input and creates two output files containing features, one for the training set and the other for the prediction set.

Command-line parameters:
* `--rdsfile` : full path to the RDS file created using R code.
* `--trainfile` : full path to the output pickle file for the training set.
* `--predictfile` : full path to the output pickle file for the prediction set.

Example command:

```
CreateFeatureFilesFromRDS.py --rdsfile /home/oleg/workspace/metap/data/input/PS118220.rds --trainfile results/PS118220/PS118220_train.pkl --predictfile results/PS118220/PS118220_predict.pkl
```

### <a name="CompareFeatures"/>Compare features generated by R and Python
`CompareRandPythonFeatureSet.py` uses features pickle files created by `CreateFeatureFilesFromRDS.py` and `GenTrainingAndTestFeatures.py` and finds the features which have different values.

Command-line parameters:
* `--pythonfile` : full path to the features picke file created by `GenTrainingAndTestFeatures.py`.
* `--rfile` : full path to the features picke file created by `CreateFeatureFilesFromRDS.py`.
* `--decimalplace` : Feature values genereated by R and Python might have different number of digits after the decimal place. Use this parameter to specify the number of digits after the decimal place. Program rounds the floating point numbers to the given decimal place. The default value is 2.


Example command:

```
CompareRandPythonFeatureSet.py --pythonfile results/ATG_NEG_NO_LINCS/atg_no_lincs_TrainingData.pkl --rfile results/ATG_NEG_NO_LINCS/ATG_NEG_NO_LINCS_train.pkl --decimalplace 3
```

### <a name="FindCommonPid"/>Find common protein ids in R and Python classification results
`FindCommonPid.py` uses classification results generated by R and Python for tranining and prediction sets and finds the common protein ids in top 'N' (100,200,...1000) proteins in training and prediction sets. Proteins are first sorted by their predicted probabilities in the descending order and then common proteins are searched.

Command-line parameters:
* `--pythonTr` : full path to the classification results file for training set created by Python code `TrainModelML.py'.
* `--pythonPr` : full path to the classification results file for prediction set created by Python code `PredictML.py'.
* `--rTr` : full path to the classification results file for training set created by R code.
* `--rPr` : full path to the classification results file for prediction set created by R code.
* `--imgfile` : full path to the output file to save the plot.
* `--maxlimit` : Maximum number of proteins to compare. Default value is 1000.


Example command:

```
FindCommonPid.py --pythonTr results/ATG_NEG/classificationResults_XGBCrossValPred.tsv --pythonPr results/ATG_NEG/classificationResults_XGBPredict.tsv --rTr /home/oleg/workspace/metap/data/output/ATG_NEG/train.pred.tsv --rPr /home/oleg/workspace/metap/data/output/ATG_NEG/blind.pred.tsv --imgfile results/ATG_NEG/common_pid.png --maxlimit 1000
```

### <a name="FindCorrelationCoefficient"/>Find Pearson correlation coefficient using R and Python classification results
`FindCorrelation.py` uses classification results generated by R and Python for tranining/prediction sets and computes Pearson correlation coefficient using the ML predicted probabilites of proteins. Proteins are first sorted by their predicted probabilities in the descending order and then top 'N' proteins are selected to determine the correlation between R and Python results.

Command-line parameters:
* `--pythonfile` : full path to the classification results file created by Python code `TrainModelML.py'.
* `--rfile` : full path to the classification results file created by R code.
* `--tsvfile` : full path to the output file where common proteins will be saved.
* `--maxlimit` : Maximum number of proteins to compare. Default value is 100.


Example command:

```
FindCorrelation.py --pythonfile results/ATG_NEG/classificationResults_XGBPredict.tsv --rfile /home/oleg/workspace/metap/data/output/ATG_NEG/blind.pred.tsv --tsvfile results/ATG_NEG/common_pid_top100.tsv --maxlimit 1000
```

## <a name="Notes"/>Notes

* The code currently assumes that all nodes are unique, that proteins are integer IDs, and the only ints in the graph. 
* New data sources can be supported by adding new Adapter class in `ProteinGraphML/DataAdapter/`.
* New ML procedures may be added to `ProteinGraphML/MLTools/Procedures/`.

Workflow overview diagram:

> <img src="doc/MPML_Workflow_Overview.png" height="400">
