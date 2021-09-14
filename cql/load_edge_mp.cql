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

////
//Delete duplicates:
//MATCH (s)-[r:GO]->(e) WITH s, e, type(r) AS t, tail(collect(r)) AS c foreach(x IN c | DELETE x)
