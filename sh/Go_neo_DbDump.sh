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
#sudo -u neo4j neo4j-admin load --database=${DBNAME} --from=$DATADIR/mpml_neo4j_${DBNAME}.dump
#[--force]
