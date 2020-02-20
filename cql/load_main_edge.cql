// Loads nodes and edges from TSV produced by BuildKG.py, copied to
// default import dir /var/lib/neo4j/import/.
////
// See https://neo4j.com/docs/getting-started/current/cypher-intro/load-csv/
// CALL dbms.security.createUser('jjyang', 'assword')
// In Community Edition, all users have admin privileges.
////


////
// EDGES ("relationships")
////
// ISSUE: Can we load all relationships in one command? Can we use row.label to
// define the relationship type? Apparently "CREATE (s)-[:row.label " not ok.
////
// ISSUE: Why are there protein-protein edges?? Bug?
////
// NOTE: Cypher regex matches ENTIRE string.
////
//Debugging:
//LOAD CSV WITH HEADERS FROM "file:///ProteinGraphML/kg.tsv"
//AS row FIELDTERMINATOR '\t' WITH row
//MATCH (s {ID:row.sourceId}), (t {ID:row.targetId})
//WHERE row.node_or_edge = 'edge'
//RETURN row.sourceId, row.targetId, s.ID, t.ID ;
////
//USING PERIODIC COMMIT 100
//LOAD CSV WITH HEADERS FROM "file:///ProteinGraphML/kg.tsv"
//AS row FIELDTERMINATOR '\t' WITH row
//MATCH (s {ID:row.sourceId}), (t {ID:row.targetId})
//WHERE row.node_or_edge = 'edge'
//AND NOT (row.sourceId =~ '[0-9]+' AND row.targetId =~ '[0-9]+')
//CREATE (s)-[:Unknown]->(t) ;
//
USING PERIODIC COMMIT 100
LOAD CSV WITH HEADERS FROM "file:///ProteinGraphML/kg.tsv"
AS row FIELDTERMINATOR '\t' WITH row
MATCH (s {ID:row.sourceId}), (t {ID:row.targetId})
WHERE row.node_or_edge = 'edge'
AND toString(row.sourceId) =~ '[0-9]+'
AND SUBSTRING(row.targetId, 0, 3) = 'GO:'
CREATE (s)-[:GO]->(t) ;

USING PERIODIC COMMIT 100
LOAD CSV WITH HEADERS FROM "file:///ProteinGraphML/kg.tsv"
AS row FIELDTERMINATOR '\t' WITH row
MATCH (s {ID:row.sourceId}), (t {ID:row.targetId})
WHERE row.node_or_edge = 'edge'
AND toString(row.sourceId) =~ '[0-9]+'
AND SUBSTRING(row.targetId, 0, 5) = 'R-HSA'
CREATE (s)-[:REACTOME]->(t) ;

USING PERIODIC COMMIT 100
LOAD CSV WITH HEADERS FROM "file:///ProteinGraphML/kg.tsv"
AS row FIELDTERMINATOR '\t' WITH row
MATCH (s {ID:row.sourceId}), (t {ID:row.targetId})
WHERE row.node_or_edge = 'edge'
AND toString(row.sourceId) =~ '[0-9]+'
AND SUBSTRING(row.targetId, 0, 3) = 'hsa'
CREATE (s)-[:KEGG]->(t) ;

USING PERIODIC COMMIT 100
LOAD CSV WITH HEADERS FROM "file:///ProteinGraphML/kg.tsv"
AS row FIELDTERMINATOR '\t' WITH row
MATCH (s {ID:row.sourceId}), (t {ID:row.targetId})
WHERE row.node_or_edge = 'edge'
AND toString(row.sourceId) =~ '[0-9]+'
AND SUBSTRING(row.targetId, 0, 3) = 'IPR'
CREATE (s)-[:INTERPRO]->(t) ;

//USING PERIODIC COMMIT 100
//LOAD CSV WITH HEADERS FROM "file:///ProteinGraphML/kg.tsv"
//AS row FIELDTERMINATOR '\t' WITH row
//MATCH (s {ID:row.sourceId}), (t {ID:row.targetId})
//WHERE row.node_or_edge = 'edge'
//AND SUBSTRING(row.sourceId, 0, 3) = 'MP:'
//AND SUBSTRING(row.targetId, 0, 3) = 'MP:'
//CREATE (s)-[:MP_Ontology]->(t) ;
//
//USING PERIODIC COMMIT 100
//LOAD CSV WITH HEADERS FROM "file:///ProteinGraphML/kg.tsv"
//AS row FIELDTERMINATOR '\t' WITH row
//MATCH (s {ID:row.sourceId}), (t {ID:row.targetId})
//WHERE row.node_or_edge = 'edge'
//AND SUBSTRING(row.sourceId, 0, 3) = 'MP:'
//AND toString(row.targetId) =~ '[0-9]+'
//CREATE (s)-[:MP_Annotation]->(t) ;

// Report relationship count:
MATCH ()-[r]-() RETURN COUNT(r) AS RELATIONSHIPS ;
