////
// EDGES ("relationships")
////
// NOTE: LOAD CSV vals read as strings.
// NOTE: Cypher regex matches ENTIRE string.
////

USING PERIODIC COMMIT 100
LOAD CSV WITH HEADERS FROM "file:///ProteinGraphML/kg.tsv"
AS row FIELDTERMINATOR '\t' WITH row
MATCH (s {ID:row.sourceId}), (t {ID:row.targetId})
WHERE row.node_or_edge = 'edge'
AND row.sourceId =~ '[0-9]+'
AND SUBSTRING(row.targetId, 0, 3) = 'IPR'
CREATE (s)-[:INTERPRO]->(t) ;
