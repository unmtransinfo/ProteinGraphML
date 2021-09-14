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

//BUG? Why a few GO->Protein edges, but mostly Protein->GO?

USING PERIODIC COMMIT 100
LOAD CSV WITH HEADERS FROM "file:///ProteinGraphML/kg.tsv"
AS row FIELDTERMINATOR '\t' WITH row
MATCH (s {ID:row.sourceId}), (t {ID:row.targetId})
WHERE row.node_or_edge = 'edge'
AND SUBSTRING(row.targetId, 0, 3) = 'GO:'
AND row.sourceId =~ '[0-9]+'
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
AND SUBSTRING(row.sourceId , 0, 3) = 'GO:'
AND row.targetId =~ '[0-9]+'
CREATE (s)-[:GO]->(t) ;

