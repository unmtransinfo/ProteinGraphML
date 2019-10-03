# ProteinGraphML

Protein Graph in Python for MetaPath-ML and for any other graph machine learning
needs. Comes with machine learning models in Python.

## Table of Contents  

* [Dependencies](#Dependencies)
* [Background](#Background)
* [Database Integration](#Database)  
* [Machine Learning](#MachineLearning)  
* [Visualization](#Vis)  
* [Pipeline to run script](#Pipeline)
* [Steps to run script](#Steps)



## <a name="Dependencies"/>Dependencies

* `xgboost`, `scikit-learn`, `networkx`, `pandas`, `pony`, `matplotlib`
*  PostgreSQL database `metap` accessible.


NOTE: to begin add your DB creds to the DBcreds.yaml file, without a database you
cannot run the 'BuildKG_OlegDb.py' code.

That will build a graph on which metapath ML features are created

You can use the ML Example notebook with just a graph, but you will need to generate it before hand



## <a name="Background"/>Background:

The goal of this software is to enable the user to perform machine learning on disease associations.
 This repo approaches this as an ML / engineering problem first. Thus, the code is designed to operate on structure, not on data sources specifically. Hopefully this approach is helpful for future work.

The current system is designed for mammalian phenotype data assuming a human-to0mouse map, but what the data represents is not important.

There are two parts to the pipe line, first we need to take data (relational, and convert this to a graph where we will synthesize new features, and perform visualization).

This is done using networkx, and objects called “adapters”. We can create adapters which return data frames, then by mapping out the edges we’ve specified, the adapters can create edges. These edges are chained together to create a graph base, which is just a networkX graph.

Below is a diagram of the major layout of how this works:
![alt text](MetapathDiagram.png)

## <a name="Database"/>Database:

As mentioned above, if you would like to pull new data into the ProteinGraph system, you can add new adapters, this will prevent you from needing to edit any of the graph code or machine learning code, instead you can merely define a new Adapter class in: <br>
`ProteinGraphML/DataAdapter/`
<br>

### Note:

<i>The graph currently makes assumptions that all nodes are unique. Proteins are integer IDs, these are the only Ints in the graph, the current protein logic counts on this being true, so make sure you do not violate this, otherwise calculations may be off!</i>


## <a name="MachineLearning"/>Machine Learning:

After a graph has been generated, you can run machine learning on a given set of
labels, or on a given disease by using RunML.py.

The script can be run as:<br>
`RunML.py <PROCEDURE> --disease <DISEASE STRING> | --file <FILE WITH LABELS>`<br>


This script will automatically generate a set of features... which you can expose to any "Procedure". These are machine learning functions which you can just add to <br>
`ProteinGraphML/MLTools/Procedures/`<br>

Then to run a given Procedure, in this case `XGBCrossValPred`, with a given disease exposed in graph we can use:<br>
`$ RunML.py XGBCrossValPred --disease MP_0000180`<br>
(this will create features for MP_0000180, and then push that data to the procedure `XGBCrossValPred`)

We can also use a file of labels as our input:
For example:<br>
`RunML.py XGBCrossValPred --file exampleFile`
(the labels need to be in the form of a pickled dictionary)

## <a name="Vis"/>Visualization:
Scripts for visualization can be found in: <br>ProteinGraphML/Analysis/, which contains code for graph generation and for creating charts for feature visualization. Visualize has code for creating HTML graphs, and featureLabel has code for taking a dictionary of feature importance, and giving it human readable labels.
( this has old code I wrote as well, which is not integrated w/ the current system but might be useful as you integrate new parts of the system)

Also, there is an included MakeVis.py script, which can autogenerate HTML graphs given feature importance. This is an experimental file, so you may need to edit it a bit to get it to work for all situations.


## <a name="Pipeline"/>Pipeline:
To run a completed pipeline, you can use `BuildKG_OlegDb.py` which will generte
a graph, and then you can use `ML_Example.py`, or `RunML.py` to generate a set of results. Results will be recorded in `ProteinGraphML/results`. (Some example results have already been created). You can find the code to change settings for models in `ProteinGraphML/MLTools/Models`.



## <a name="Static"/>Static Features:
To generate static features (features which don't use metapaths), there is an attached R script which you can use `ProteinGraphML/MLTools/StaticFeatures/staticFiles.R` which will generte four CSV files. One for ccle,gtex,lincs, and hpa, based on Oleg's transformations and his database.
Once you run this script, all you need to do is pickle the four csv files, and then
you can use them w/ as features. For example in RunML.py (line ~75): <br>

`nodes = [ProteinInteractionNode,KeggNode,ReactomeNode,GoNode,InterproNode]`<br>
`staticFeatures = [] becomes -> ["gtex","lincs","ccle","hpa"]`<br>
`trainData = metapathFeatures(disease,currentGraph,nodes,staticFeatures).fillna(0)`<br>


This will auto load all of the static features and bind them to your data.

## <a name="Steps"/>Steps to run ProteinGraphML codes:
The codes of ProteinGraphML must be executed in the following order to avoid errors:

__1. `BuildKG_OlegDb.py`:__  Run first, to generate a knowledge graph required to run ML codes.

___KG produced may be re-used for multiple ML models. Re-run only required if database updated.___

```
BuildKG_OlegDb.py
```

__2. `PickleTrainingset.py`:__	This program generates a "pickle" dictionary that
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

```
PickleTrainingset.py --file filename.xlsx
PickleTrainingset.py --file filename.txt

E.g.
PickleTrainingset.py --file diabetes_pid.txt --symbol_or_pid 'pid'
PickleTrainingset.py --file 125853.rds
PickleTrainingset.py --file diabetes.xlsx
```

__3. `RunML.py`:__  This is the machine learning code that uses XGboost model
to classify the data using 5-fold cross-validation. It also generates a list of
important features used by the classification model.

Command line parameters:

* `procedure` (positional parameter):
   * `XGBCrossValPred` :  5-fold cross-validation for one iteration.
   * `XGBCrossVal` : 5-fold cross-validation for multiple iterations.
* `--disease` : Use with Mammalian Phenotype ID, e.g. MP_0000180.
* `--file` : Pickled training set file, produced by `PickleTrainingset.py`.

E.g. 
```
RunML.py XGBCrossVal --file 144700.pkl
RunML.py XGBCrossValPred --file 144700.pkl
RunML.py XGBCrossVal --disease MP_0000180
RunML.py XGBCrossValPred --disease MP_0000180
```

__4. `MakeVis.py`:__  Generates HTML/JS files for visualization, via web browser.

Command-line parameters:

* `--disease` pickled features file produced by `RunML.py`, e.g. diabetes.pkl.
* `--dir` : dir containing file (default: results/XGBFeatures/).
* `--num` : number of top important features selected.

E.g. 

```
MakeVis.py --disease MP_0000180.pkl --num 2
```
