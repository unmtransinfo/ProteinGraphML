#!/bin/bash
###
#
function MessageBreak {
  printf "============================================\n"
  printf "=== [%s] %s\n" "$(date +'%Y-%m-%d:%H:%M:%S')" "$1"
}
#
T0=$(date +%s)
#
MessageBreak "STARTING $(basename $0)"
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
if [ "$(which cypher-shell)" ]; then
	CYSH="$(which cypher-shell)"
else
	echo "ERROR: cypher-shell not found."
	exit
fi
printf "CYSH = %s\n" "$CYSH"
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
#$CYSH "CALL db.constraints() YIELD name AS constraint_name DROP CONSTRAINT constraint_name"
#
# Delete all:
$CYSH 'MATCH (n) DETACH DELETE n'
#
###
MessageBreak "LOAD NODES:"
$CYSH <cql/load_main_node.cql
#
MessageBreak "LOAD EDGES (STRING):"
$CYSH <cql/load_edge_string.cql
MessageBreak "LOAD EDGES (REACTOME):"
$CYSH <cql/load_edge_reactome.cql
MessageBreak "LOAD EDGES (INTERPRO):"
$CYSH <cql/load_edge_interpro.cql
MessageBreak "LOAD EDGES (KEGG):"
$CYSH <cql/load_edge_kegg.cql
MessageBreak "LOAD EDGES (GO):"
$CYSH <cql/load_edge_go.cql
MessageBreak "LOAD EDGES (MP):"
$CYSH <cql/load_edge_mp.cql
#
#$CYSH <cql/load_extras.cql
#$CYSH <cql/db_describe.cql
#
###
# Delete edges with:
# cypher-shell 'MATCH ()-[r]-() DELETE r'
# Delete all with:
# cypher-shell 'MATCH (n) DETACH DELETE n'
###
#
MessageBreak "DONE $(basename $0)"
#
