// Loads nodes and edges from TSV produced by BuildKG.py, copied to
// default import dir /var/lib/neo4j/import/.
////
// See https://neo4j.com/docs/getting-started/current/cypher-intro/load-csv/
// CALL dbms.security.createUser('www', 'foobar')
// In Community Edition, all users have admin privileges.
////


////
// NODES
// Protein
// KEGG
// GO

// Constraints.  Index auto-generated with uniqueness constraint.
CREATE CONSTRAINT ON (n:Protein) ASSERT n.ID IS UNIQUE ;
CREATE CONSTRAINT ON (n:KEGG) ASSERT n.ID IS UNIQUE ;
CREATE CONSTRAINT ON (n:GO) ASSERT n.ID IS UNIQUE ;
CREATE CONSTRAINT ON (n:REACTOME) ASSERT n.ID IS UNIQUE ;
CREATE CONSTRAINT ON (n:INTERPRO) ASSERT n.ID IS UNIQUE ;
CALL db.constraints ;
// How to drop:
// DROP CONSTRAINT constraint_name ;

// Protein:
LOAD CSV WITH HEADERS FROM "file:///ProteinGraphML/kg.tsv"
AS row FIELDTERMINATOR '\t' WITH row
WHERE row.node_or_edge = 'node' AND row.class = 'PROTEIN' AND row.id IS NOT NULL
CREATE (g:Protein { ID:row.id, Name:trim(row.name)}) ;

// KEGG:
LOAD CSV WITH HEADERS FROM "file:///ProteinGraphML/kg.tsv"
AS row FIELDTERMINATOR '\t' WITH row
WHERE row.node_or_edge = 'node' AND row.class = 'KEGG'
CREATE (g:KEGG { ID:row.id, Name:trim(row.name)}) ;

// GO:
LOAD CSV WITH HEADERS FROM "file:///ProteinGraphML/kg.tsv"
AS row FIELDTERMINATOR '\t' WITH row
WHERE row.node_or_edge = 'node' AND row.class = 'GO'
CREATE (g:GO { ID:row.id, Name:trim(row.name)}) ;

// REACTOME:
LOAD CSV WITH HEADERS FROM "file:///ProteinGraphML/kg.tsv"
AS row FIELDTERMINATOR '\t' WITH row
WHERE row.node_or_edge = 'node' AND row.class = 'REACTOME'
CREATE (g:REACTOME { ID:row.id, Name:trim(row.name)}) ;

// INTERPRO:
LOAD CSV WITH HEADERS FROM "file:///ProteinGraphML/kg.tsv"
AS row FIELDTERMINATOR '\t' WITH row
WHERE row.node_or_edge = 'node' AND row.class = 'INTERPRO'
CREATE (g:INTERPRO { ID:row.id, Name:trim(row.name)}) ;

// MP:
LOAD CSV WITH HEADERS FROM "file:///ProteinGraphML/kg.tsv"
AS row FIELDTERMINATOR '\t' WITH row
WHERE row.node_or_edge = 'node' AND row.class = 'MP'
CREATE (g:MP { ID:row.id, Name:trim(row.name)}) ;

// Report node counts:
MATCH (n) RETURN COUNT(n) AS N_NODES_ALL ;
MATCH (n:Protein) RETURN COUNT(n) AS N_NODES_Protein ;
MATCH (n:GO) RETURN COUNT(n) AS N_NODES_GO ;
MATCH (n:REACTOME) RETURN COUNT(n) AS N_NODES_REACTOME ;
MATCH (n:INTERPRO) RETURN COUNT(n) AS N_NODES_INTERPRO ;

////
// Indexes. AUTO-GENERATED WITH UNIQUENESS CONSTRAINTS.
// Community:
//CALL db.createIndex("Protein_ID_Index", ["Protein"], ["ID"], "native-btree-1.0") ;
//CALL db.createIndex("KEGG_ID_Index", ["KEGG"], ["ID"], "native-btree-1.0") ;
//CALL db.createIndex("GO_ID_Index", ["GO"], ["ID"], "native-btree-1.0") ;
//CALL db.createIndex("REACTOME_ID_Index", ["REACTOME"], ["ID"], "native-btree-1.0") ;
//CALL db.createIndex("INTERPRO_ID_Index", ["INTERPRO"], ["ID"], "native-btree-1.0") ;
// How to drop:
//DROP INDEX ON:Protein(ID) ;
