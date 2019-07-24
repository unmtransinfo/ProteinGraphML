# ProteinGraphML

## Dependencies

* `xgboost`, `scikit-learn`, `networkx`, `pandas`, `pony`, `matplotlib`
*  PostgreSQL database `metap` accessible.



NOTE:: to begin add your DB creds to the DBcreds.yaml file, without a database you cannot run the 'Build_Graph' notebook

That will build a graph on which metapath ML features are created 

You can use the ML Example notebook with just a graph, but you will need to generate it before hand

##### Table of Contents  
[Background](#Background)  
[Machine Learning](#MachineLearning)  

<br><br><br>
<a name="Background"/>
## Background: 

The goal of this software is to enable the user to perform machine learning on disease associations. 
 This repo approaches this as an ML / engineering problem first. Thus, the code is designed to operate on structure, not on data sources specifically. Hopefully this approach is helpful for future work. 

The current system is designed for mammalian phenotype data assuming a human-to0mouse map, but what the data respresnts is not important.

There are two parts to the pipe line, first we need to take data (relational, and convert this to a graph where we will synthesize new features, and perform visualization). 

This is done using networkx, and objects called “adapters”. We can create adapters which return data frames, then by mapping out the edges we’ve specified, the adapters can create edges. These edges are chained together to create a graph base, which is just a networkX graph. 

Below is a diagram of the major layout of how this works:
![alt text](https://github.com/unmtransinfo/ProteinGraphML/blob/master/MetapathDiagram.png)

<br><br><br>
<a name="MachineLearning"/>
## Machine Learning instructions: 

After a graph has been generated, you can run machine learning on a given set of labels, or on a given disease by using scriptML.py.

Input for this script will be:
python scriptML.py <PROCEDURE> --disease <DISEASE STRING> | --file <FILE WITH LABELS>


This script will automatically generate a set of features... which you can expose to any "Procedure". These are machine learning functions which you can just add to 
MLTools/Procedures/

Then to run a given Procedure with a given disease exposed in the graph we can use:
python scriptML.py XGBCrossValPred --disease MP_0000180
(this will create features for MP_0000180, and then push that data to the procedure 'XGBCrossValPred')

We can also use a file of labels as our input:
For example:
python scriptML.py XGBCrossValPred --file ATG_KEGG
(the labels need to be in the form of a pickled dictionary)










