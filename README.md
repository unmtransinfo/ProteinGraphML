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
   * [Static Features](#HowtoStaticFeatures)  
   * [Pickle Training Set](#HowtoPickleTrainingset)  
   * [Run ML Procedure](#HowtoRunML)  
   * [Visualization](#HowtoVis)  

## <a name="Dependencies"/>Dependencies

* R 3.5+
* R packages: `data.table`, `Matrix`, `RPostgreSQL`
* Python 3.4+
* Python packages: `xgboost`, `scikit-learn`, `networkx`, `pandas`, `pony`, `matplotlib`
* PostgreSQL database `metap`.

## <a name="Howto"/>How to run the Workflow:

The command-line programs of ProteinGraphML must be executed in the following order.
However, the __BuildKG__ and __StaticFeatures__ are one-time steps, re-useable for
multiple ML models. Re-run only required if database updated.


### <a name="HowtoBuildKG"/>Build KG

`BuildKG_OlegDb.py`, from the relational db, generates a knowledge graph,
saved as a pickled `networkX` graph.  Via the adaptor and `pony` object-relational model
(ORM), nodes and edges are queried from the db to comprise the graph.

```
BuildKG_OlegDb.py
```


### <a name="HowtoStaticFeatures"/>Static Features

To generate static features (not metapath-based), use R script
`ProteinGraphML/MLTools/StaticFeatures/staticFiles.R` to generate CSV files, for ccle,
gtex, lincs and hpa.  Then pickle pandas dataframes from the four csv
files, for use by `RunML.py`.


### <a name="HowtoPickleTrainingset"/>Pickle Training Set

`PickleTrainingset.py` generates a "pickle" dictionary that
contains protein_ids for both class 'True' and 'False'. The "pickle" dictionary is needed
if you are running ML codes for a disease that does not have MP_TERM_ID. Also, if you
have gene symbols instead of protein_ids for a disease, this code fetches the corresponding
protein_id for each symbol from the database and generates the "pickle" dictionary.

Command line parameters:

* `--file` : File that contains protein_ids/symbols and labels for a given disease, with extension (.txt|.xlsx|.rds).
* `--dir` : directory where data files are found (default: DataForML).
* `--symbol_or_pid` : "symbol" or "pid" (default: symbol).

If the file is a spreadsheet, the header should have "Protein_id Label" or "Symbol Label".
If the file is a text file, the Protein_id/symbol and
Label should be comma-separated. There should not be any header in the text file. If the
file is an RDS file, the parameter 'symbol'  can be omitted. Use one of the following,
to run this program.

Example commands:

```
PickleTrainingset.py --file diabetes_pid.txt --symbol_or_pid 'pid'
PickleTrainingset.py --file 125853.rds
PickleTrainingset.py --file diabetes.xlsx
```


### <a name="HowtoRunML"/>Run ML Procedure

`RunML.py`, from the input disease or training set and KG, generates feature vectors,
and executes specified ML procedure.  The procedure `XGBCrossVal` uses
XGBoost, trains a model, with cross-validation and grid-search parameter optimization,
generates a list of important features used by the classification model,
and generates results for predictions on all proteins.  

Command line parameters:

* `procedure` (positional parameter):
   * `XGBCrossValPred` :  5-fold cross-validation for one iteration.
   * `XGBCrossVal` : 5-fold cross-validation for multiple iterations.
* `--disease` : Use with Mammalian Phenotype ID, e.g. MP_0000180.
* `--file` : Pickled training set file, produced by `PickleTrainingset.py`.
* `--kgfile` : Pickled KG file, produced by `BuildKG_OlegDb.py` (default: ProteinDisease_GRAPH.pkl).

Example commands:

```
RunML.py XGBCrossVal --file 144700.pkl
RunML.py XGBCrossValPred --file 144700.pkl
RunML.py XGBCrossVal --disease MP_0000180
RunML.py XGBCrossValPred --disease MP_0000180
```

Results will be saved in `ProteinGraphML/results`. See logs for specific
subdirectories and output files, including:

* Saved XGBoost model (.model).
* Predictions with probabilities for all proteins (.tsv, .xlsx).
* Feature importance lists (.tsv, .xlsx).


### <a name="HowtoVis"/>Visualize _(optional)_

`MakeVis.py` generates HTML/JS files, with feature importance, for visualization
via web browser.  `ProteinGraphML/Analysis/` contains code for graph and feature
visualization. Visualize has code for creating HTML/JS graphs, and featureLabel has
code for taking a dictionary of feature importance, and giving it human readable labels.

Command-line parameters:

* `--disease` pickled features file produced by `RunML.py`, e.g. diabetes.pkl.
* `--dir` : dir containing file (default: results/XGBFeatures/).
* `--num` : number of top important features selected.
* `--kgfile` : Pickled KG file, produced by `BuildKG_OlegDb.py` (default: ProteinDisease_GRAPH.pkl).

E.g.

```
MakeVis.py --disease MP_0000180.pkl --num 2
```

## <a name="Notes"/>Notes

* The current system is designed for mammalian phenotype data assuming a human-to-mouse map.
* To begin add your DB creds to the DBcreds.yaml file.
* The graph currently makes assumptions that all nodes are unique. Proteins are integer IDs, these are the only Ints in the graph, the current protein logic counts on this being true, so make sure you do not violate this, otherwise calculations may be off!
* New data sources can be supported via adding new Adapter class in `ProteinGraphML/DataAdapter/`.
* New machine learning procedures may be added to `ProteinGraphML/MLTools/Procedures/`.

> <img src="MetapathDiagram.png" height="400">
