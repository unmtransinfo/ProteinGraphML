# ProteinGraphML



NOTE:: to begin add your DB creds to the DBcreds.yaml file, without a database you cannot run the 'Build_Graph' notebook

That will build a graph on which metapath ML features are created 

You can use the ML Example notebook with just a graph, but you will need to generate it before hand


Goal of the process is to perform machine learning on disease associations. 
 This repo approaches this as an ML / engineering problem first. Thus, the code is designed to operate on structure, not on data sources specifically. Hopefully this approach is helpful for future work. 

The current system is designed for mammalian phenotype data assuming a human-to0mouse map, but what the data respresnts is not important.

There are two parts to the pipe line, first we need to take data (relational, and convert this to a graph where we will synthesize new features, and perform visualization). 

This is done using networkx, and objects called “adapters”. We can create adapters which return data frames, then by mapping out the edges we’ve specified, the adapters can create edges. These edges are chained together to create a graph base, which is just a networkX graph. 

Below is a diagram of the major layout of how this works:
![alt text](https://github.com/unmtransinfo/ProteinGraphML/blob/master/MetapathDiagram.png)





