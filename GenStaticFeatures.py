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
  logging.info("GTEX: rows: {0}; cols: {1}".format(gtex.shape[0], gtex.shape[1]))
  gtex.to_csv("gtex.tsv", "\t", index=False)

  hpa = staticData.hpa(dbad)
  logging.info("HPA: rows: {0}; cols: {1}".format(hpa.shape[0], hpa.shape[1]))
  hpa.to_csv("hpa.tsv", "\t", index=False)

  ccle = staticData.ccle(dbad)
  logging.info("CCLE: rows: {0}; cols: {1}".format(ccle.shape[0], ccle.shape[1]))
  ccle.to_csv("ccle.tsv", "\t", index=False)

  lincs = staticData.lincs(dbad)
  logging.info("LINCS: rows: {0}; cols: {1}".format(lincs.shape[0], lincs.shape[1]))
  lincs.to_csv("lincs.tsv", "\t", index=False)
