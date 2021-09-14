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

