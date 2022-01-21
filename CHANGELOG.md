New files that were added. 

# KGutilities.py 

* Containing methods for exporting the NetworkX knowledge graph.
* Method to create Neo4j compliant cypher queries to generate the nodes and their relationships. 

# loadNeo4j.py 

* Upload Cypher Queries to a Neo4j instance. 
* This is a design question. 
* This method can be added to the **BuildKG.py** command-line options. 

### Current Problems 

* Can not figure out how to translate the edges in the network to Neo4j relationships.
* The current data format for edges looks like this: 
  * ```
    MP:0002098 -> 15263
    MP:0002098 -> 17891
    MP:0002098 -> 17000
    MP:0002098 -> 2389
    MP:0002098 -> 15378
    ....
    ```
* **Question**: How can I know which numbers pertain to what node?