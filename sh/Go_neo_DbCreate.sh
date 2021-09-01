#!/bin/bash
#
cwd=$(pwd)
#
NEO4J_HOST=localhost
NEO4J_PORT=7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=assword
#
NEO4J_URL="bolt://${NEO4J_HOST}:${NEO4J_PORT}"
#
NEO_IMPORT_DIR="/var/lib/neo4j/import"
#
if [ ! -e $NEO_IMPORT_DIR ]; then
	echo "ERROR: Neo4j import dir ${NEO_IMPORT_DIR} not found."
	exit
fi
#
# Unfortunately neo4j-client and cypher-shell have different syntax.
#
if [ "$(which neo4j-client)" ]; then
	CQLAPP="neo4j-client"
elif [ "$(which cypher-shell)" ]; then
	CQLAPP="cypher-shell"
else
	echo "ERROR: Neo4j/CQL client app not found."
	exit
fi
printf "CQLAPP = %s\n" "$CQLAPP"
#
DATADIR="$NEO_IMPORT_DIR/ProteinGraphML"
if [ ! -e "$DATADIR" ]; then
	sudo -u neo4j mkdir "$DATADIR"
	sudo -u neo4j chmod g+w "$DATADIR"
fi
#
#
tsvfile="ProteinDisease_GRAPH_tcrd6110.tsv"
#
${cwd}/BuildKG.py --db tcrd --tsvfile ${cwd}/data/${tsvfile}
#
###
# Perhaps "neo4j-admin import" would be faster?
#
cp ${cwd}/data/${tsvfile} ${DATADIR}/kg.tsv
#
#$CQLAPP "CALL db.constraints() YIELD name AS constraint_name DROP CONSTRAINT constraint_name"
#
# Delete all:
$CQLAPP 'MATCH (n) DETACH DELETE n'
#
###
$CQLAPP -i cql/load_main_node.cql ${NEO4J_URL}
$CQLAPP -i cql/load_main_edge.cql ${NEO4J_URL}
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

#$CQLAPP -i cql/load_extras.cql
#$CQLAPP -i cql/db_describe.cql
#
###
# Delete edges with:
# cypher-shell 'MATCH ()-[r]-() DELETE r'
# Delete all with:
# cypher-shell 'MATCH (n) DETACH DELETE n'
###
#
