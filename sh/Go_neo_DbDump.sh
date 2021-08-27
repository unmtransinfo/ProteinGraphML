#!/bin/bash
#
DATADIR="/var/lib/neo4j/tmp"
if [ ! -e "$DATADIR" ]; then
	mkdir "$DATADIR"
fi
#
###
DBNAME="neo4j"
#
sudo systemctl -l status neo4j
sudo systemctl -l stop neo4j
#
sudo -u neo4j neo4j-admin dump --database=${DBNAME} --to=$DATADIR/mpml_neo4j_${DBNAME}.dump
#
sudo systemctl -l start neo4j
sudo systemctl -l status neo4j
#
###
# Neo4j Server: 
#sudo -u neo4j neo4j-admin load --database=${DBNAME} --from=$DATADIR/mpml_neo4j_${DBNAME}.dump
#[--force]
###
# Neo4j Desktop: 
# To restore this graph database from the dump file:
#   - Install and launch the Neo4j Desktop Client.
#   - Create a new database (do not start the database).
#   - Click on the ... then click Manage in the pop up menu, then click Open Terminal.
#   - Load the dump file with the following command, changing PATHTOFILE as needed.
#
# bin/neo4j-admin load --from=PATHTOFILE/mpml.dump --database=neo4j
#
#   - Exit terminal and click Start to start the database.
#   - Click Open to use built in Neo4j browser to query and explore the database.
###
