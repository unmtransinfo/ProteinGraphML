#!/bin/bash
#
cwd=$(pwd)
#
NEO_IMPORT_DIR="/var/lib/neo4j/import"
#
if [ ! -e $NEO_IMPORT_DIR ]; then
	echo "ERROR: Neo4j import dir ${NEO_IMPORT_DIR} not found."
	exit
fi
#
if [ "$(which cypher-shell)" ]; then
	CQLAPP="cypher-shell"
elif [ "$(which neo4j-client)" ]; then
	CQLAPP="neo4j-client"
else
	echo "ERROR: Neo4j/CQL client app not found."
	exit
fi
printf "CQLAPP = %s\n" "$CQLAPP"
#
DATADIR="$NEO_IMPORT_DIR/ProteinGraphML"
if [ ! -e "$DATADIR" ]; then
	mkdir "$DATADIR"
fi
#
#
${cwd}/BuildKG.py --db tcrd --tsvfile ${cwd}/data/kg.tsv
#
###
# Perhaps "neo4j-admin import" would be faster?
#
cp ${cwd}/data/kg.tsv ${DATADIR}
#
#$CQLAPP "CALL db.constraints() YIELD name AS constraint_name DROP CONSTRAINT constraint_name"
#
$CQLAPP -f cql/load_main_node.cql
$CQLAPP -f cql/load_main_edge.cql
#
$CQLAPP "\
USING PERIODIC COMMIT 100 \
LOAD CSV WITH HEADERS FROM \"file:///ProteinGraphML/kg.tsv\" \
AS row FIELDTERMINATOR '\t' WITH row \
MATCH (s {ID:row.sourceId}), (t {ID:row.targetId}) \
WHERE row.node_or_edge = 'edge' \
AND toString(row.sourceId) =~ '[0-9]+' \
AND SUBSTRING(row.targetId, 0, 3) = 'GO:' \
CREATE (s)-[:GO]->(t)"

#$CQLAPP -f cql/load_extras.cql
#$CQLAPP -f cql/db_describe.cql
#
###
# Delete edges with:
# cypher-shell 'MATCH ()-[r]-() DELETE r'
# Delete all with:
# cypher-shell 'MATCH (n) DETACH DELETE n'
###
#
