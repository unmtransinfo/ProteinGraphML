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


## <a name="Dependencies"/>Dependencies

* `xgboost`, `scikit-learn`, `networkx`, `pandas`, `pony`, `matplotlib`, `pyreadr`
*  PostgreSQL db `metap` accessible.
*  Db `metap` and method based on [metap](https://github.com/unmtransinfo/metap) mostly-R code originally developed by Oleg Ursu.


NOTE:: to begin add your DB creds to the DBcreds.yaml file, without a database you cannot run the `Build_Graph` notebook

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

After a graph has been generated, you can run machine learning on a given set of labels, or on a given disease by using scriptML.py.

The script can be run as:<br>
`python scriptML.py <PROCEDURE> --disease <DISEASE STRING> | --file <FILE WITH LABELS>`<br>


This script will automatically generate a set of features... which you can expose to any "Procedure". These are machine learning functions which you can just add to <br>
`ProteinGraphML/MLTools/Procedures/`<br>

Then to run a given Procedure, in this case `XGBCrossValPred`, with a given disease exposed in graph we can use:<br>
`$ python scriptML.py XGBCrossValPred --disease MP_0000180`<br>
(this will create features for MP_0000180, and then push that data to the procedure `XGBCrossValPred`)

We can also use a file of labels as our input:
For example:<br>
`python scriptML.py XGBCrossValPred --file exampleFile`
(the labels need to be in the form of a pickled dictionary)<br>
For the diseases that do not have MP_TERM_ID, train/test sets and their labels are stored in RDS files on seaborgium home/oleg/workspace/metap/data/input. Using the RDS file for a disease, scriptML.py creates a dictionary with keys True and False (e.g. {True:{1,2,3},False:{5,6}}). To run the ML scripts for such diseases, use file as an argument. E.g.<br>
`python scriptML.py XGBCrossValPred --file 1014300` or <br>
`python scriptML.py XGBCrossVal --file 1014300`

## <a name="Vis"/>Visualization:
Scripts for visualization can be found in: <br>ProteinGraphML/Analysis/, which contains code for graph generation and for creating charts for feature visualization. Visualize has code for creating HTML graphs, and featureLabel has code for taking a dictionary of feature importance, and giving it human readable labels.
( this has old code I wrote as well, which is not integrated w/ the current system but might be useful as you integrate new parts of the system)

Also, there is an included makeVis.py script, which can autogenerate HTML graphs given feature importance. This is an experimental file, so you may need to edit it a bit to get it to work for all situations.


## <a name="Pipeline"/>Pipeline:
To run a completed pipeline, you can use `Build_Graph_Example.py` which will generte a graph, and then you can use `ML_Example.py`, or `scriptML.py` to generate a set of results. Results will be recorded in `ProteinGraphML/results`. (Some example results have already been created). You can find the code to change settings for models in `ProteinGraphML/MLTools/Models`.



## <a name="Static"/>Static Features:
To generate static features (features which don't use metapaths), there is an attached R script which you can use `ProteinGraphML/MLTools/StaticFeatures/staticFiles.R` which will generte four CSV files. One for ccle,gtex,lincs, and hpa, based on Oleg's transformations and his database.
Once you run this script, all you need to do is pickle the four csv files, and then you can use them w/ as features. For example in scriptML.py (line ~75): <br>

`nodes = [ProteinInteractionNode,KeggNode,ReactomeNode,GoNode,InterproNode]`<br>
`staticFeatures = [] becomes -> ["gtex","lincs","ccle","hpa"]`<br>
`trainData = metapathFeatures(disease,currentGraph,nodes,staticFeatures).fillna(0)`<br>


This will auto load all of the static features and bind them to your data.
