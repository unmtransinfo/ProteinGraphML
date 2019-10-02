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

__1. `BuildKG_OlegDb.py`:__  Run this program before running any other program as it generates a graph that is required to run ML codes.
```
BuildKG_OlegDb.py
```

__2. `PickleTrainingset.py`:__  This program generates a "pickle" dictionary that contains protein_ids for both class 'True' and 'False'. The "pickle" dictionary is needed if you are running ML codes for a disease that does not have MP_TERM_ID. Also, if you have symbols instead of protein_ids for a disease, this code fetches the corresponding protein_id for each symbol from the database and generates the "pickle" dictionary. This program needs 2 command line parameters -  'file' and 'symbol'. Parameter 'file' is used to specify the name of the file that contains protein_ids/symbols and labels for a given disease. Parameter 'symbol' is used to specify whether or not the file contains symbols. If the file contains symbols, the value of the parameter 'symbol' is set to Y, otherwise, N. If the file is a spreadsheet, the header should have "Protein_id	 Label " or "Symbol    Label". If the file is a text file, the Protein_id/symbol and  Label should be comma-separated. There should not be any header in the text file. If the file is an RDS file, the parameter 'symbol'  can be omitted. Use one of the following, to run this program.
```
PickleTrainingset.py --file filename.xlsx --symbol Y
PickleTrainingset.py --file filename.xlsx --symbol N
PickleTrainingset.py --file filename.txt --symbol Y
PickleTrainingset.py --file filename.txt --symbol N
PickleTrainingset.py --file RDS_file

E.g.
PickleTrainingset.py --file 125853
PickleTrainingset.py --file diabetes.xlsx --symbol Y
```

__3. `RunML.py`:__  This is the machine learning code that uses XGboost model to classify the data using 5-fold cross-validation. It also generates a list of important features used by the classification model. If a disease has MP_TERM_ID, run this program using parameter 'disease', otherwise, run using parameter 'file'. Also, use parameter 'XGBCrossValPred' if you want to use run 5-fold cross-validation for one iteration. Use parameter 'XGBCrossVal' if you want to run 5-fold cross-validation for multiple iterations. 
```
RunML.py XGBCrossValPred --file filename
RunML.py XGBCrossVal --file filename
RunML.py XGBCrossValPred --disease diseasename
RunML.py XGBCrossVal --disease diseasename

E.g. 
RunML.py XGBCrossVal --file 144700
RunML.py XGBCrossValPred --disease MP_0000180
```

__4. `MakeVis.py`:__  This program generates HTML files containing JS/HTML codes for visualization. This program needs 2 command-line parameters: 'disease' and 'num'. Parameter 'num' represents the number of top important features that need to be selected. Parameter 'disease' is used for selecting a disease. If the disease is not present in the graph, this code will return an error. 
```
MakeVis.py --disease diseasename --num numberOfImportantFeatures

E.g. 
MakeVis.py --disease MP_0000180 --num 2
```
