////
// EDGES ("relationships")
////
// NOTE: LOAD CSV vals read as strings.
// NOTE: Cypher regex matches ENTIRE string.
////
//Debugging:
//LOAD CSV WITH HEADERS FROM "file:///ProteinGraphML/kg.tsv"
//AS row FIELDTERMINATOR '\t' WITH row
//MATCH (s {ID:row.sourceId}), (t {ID:row.targetId})
//WHERE row.node_or_edge = 'edge'
//RETURN row.sourceId, row.targetId, s.ID, t.ID ;
////
//Protein-Protein (STRING)
USING PERIODIC COMMIT 100
LOAD CSV WITH HEADERS FROM "file:///ProteinGraphML/kg.tsv"
AS row FIELDTERMINATOR '\t' WITH row
MATCH (s {ID:row.sourceId}), (t {ID:row.targetId})
WHERE row.node_or_edge = 'edge'
AND row.sourceId =~ '[0-9]+'
AND row.targetId =~ '[0-9]+'
CREATE (s)-[:STRING]->(t) ;

USING PERIODIC COMMIT 100
LOAD CSV WITH HEADERS FROM "file:///ProteinGraphML/kg.tsv"
AS row FIELDTERMINATOR '\t' WITH row
MATCH (s {ID:row.sourceId}), (t {ID:row.targetId})
WHERE row.node_or_edge = 'edge'
AND row.sourceId =~ '[0-9]+'
AND SUBSTRING(row.targetId, 0, 3) = 'GO:'
CREATE (s)-[:GO]->(t) ;

//BUG? Why a few GO->Protein edges, but mostly Protein->GO?
USING PERIODIC COMMIT 100
LOAD CSV WITH HEADERS FROM "file:///ProteinGraphML/kg.tsv"
AS row FIELDTERMINATOR '\t' WITH row
MATCH (s {ID:row.sourceId}), (t {ID:row.targetId})
WHERE row.node_or_edge = 'edge'
AND SUBSTRING(row.sourceId , 0, 3) = 'GO:'
AND row.targetId =~ '[0-9]+'
CREATE (s)-[:GO]->(t) ;

USING PERIODIC COMMIT 100
LOAD CSV WITH HEADERS FROM "file:///ProteinGraphML/kg.tsv\"
AS row FIELDTERMINATOR '\t' WITH row
MATCH (s {ID:row.sourceId}), (t {ID:row.targetId})
WHERE row.node_or_edge = 'edge'
AND SUBSTRING(row.targetId, 0, 3) = 'GO:'
AND toString(row.sourceId) =~ '[0-9]+'
CREATE (s)-[:GO]->(t) ;

USING PERIODIC COMMIT 100
LOAD CSV WITH HEADERS FROM "file:///ProteinGraphML/kg.tsv"
AS row FIELDTERMINATOR '\t' WITH row
MATCH (s {ID:row.sourceId}), (t {ID:row.targetId})
WHERE row.node_or_edge = 'edge'
AND row.sourceId =~ '[0-9]+'
AND SUBSTRING(row.targetId, 0, 5) = 'R-HSA'
CREATE (s)-[:REACTOME]->(t) ;

USING PERIODIC COMMIT 100
LOAD CSV WITH HEADERS FROM "file:///ProteinGraphML/kg.tsv"
AS row FIELDTERMINATOR '\t' WITH row
MATCH (s {ID:row.sourceId}), (t {ID:row.targetId})
WHERE row.node_or_edge = 'edge'
AND row.sourceId =~ '[0-9]+'
AND SUBSTRING(row.targetId, 0, 3) = 'hsa'
CREATE (s)-[:KEGG]->(t) ;

USING PERIODIC COMMIT 100
LOAD CSV WITH HEADERS FROM "file:///ProteinGraphML/kg.tsv"
AS row FIELDTERMINATOR '\t' WITH row
MATCH (s {ID:row.sourceId}), (t {ID:row.targetId})
WHERE row.node_or_edge = 'edge'
AND row.sourceId =~ '[0-9]+'
AND SUBSTRING(row.targetId, 0, 3) = 'IPR'
CREATE (s)-[:INTERPRO]->(t) ;

USING PERIODIC COMMIT 100
LOAD CSV WITH HEADERS FROM "file:///ProteinGraphML/kg.tsv"
AS row FIELDTERMINATOR '\t' WITH row
MATCH (s {ID:row.sourceId}), (t {ID:row.targetId})
WHERE row.node_or_edge = 'edge'
AND SUBSTRING(row.sourceId, 0, 3) = 'MP:'
AND SUBSTRING(row.targetId, 0, 3) = 'MP:'
CREATE (s)-[:MP_Ontology]->(t) ;
//
USING PERIODIC COMMIT 100
LOAD CSV WITH HEADERS FROM "file:///ProteinGraphML/kg.tsv"
AS row FIELDTERMINATOR '\t' WITH row
MATCH (s {ID:row.sourceId}), (t {ID:row.targetId})
WHERE row.node_or_edge = 'edge'
AND SUBSTRING(row.sourceId, 0, 3) = 'MP:'
AND row.targetId =~ '[0-9]+'
CREATE (s)-[:MP_Annotation]->(t) ;

// Report relationship count:
MATCH ()-[r]-() RETURN COUNT(r) AS RELATIONSHIPS_All ;
MATCH ()-[r:GO]-() RETURN COUNT(r) AS RELATIONSHIPS_GO ;
MATCH ()-[r:KEGG]-() RETURN COUNT(r) AS RELATIONSHIPS_KEGG ;
MATCH ()-[r:REACTOME]-() RETURN COUNT(r) AS RELATIONSHIPS_REACTOME ;
MATCH ()-[r:INTERPRO]-() RETURN COUNT(r) AS RELATIONSHIPS_INTERPRO ;
////
//Delete duplicates:
//MATCH (s)-[r:GO]->(e) WITH s, e, type(r) AS t, tail(collect(r)) AS c foreach(x IN c | DELETE x)
