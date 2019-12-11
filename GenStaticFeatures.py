#!/usr/bin/env python3
###

import sys,os,argparse,time
import logging
import pandas as pd

from ProteinGraphML.DataAdapter import OlegDB
from ProteinGraphML.MLTools.StaticFeatures import staticData

###
if __name__ == '__main__':

  t0 = time.time()
  dbad = OlegDB()

  gtex = staticData.gtex(dbad)
  logging.info("rows: {0}; cols: {1}".format(gtex.shape[0], gtex.shape[1]))
  gtex.to_csv("gtex.tsv", "\t", index=False)

  lincs = staticData.lincs(dbad)
  logging.info("rows: {0}; cols: {1}".format(lincs.shape[0], lincs.shape[1]))
  lincs.to_csv("lincs.tsv", "\t", index=False)
