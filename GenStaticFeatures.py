#!/usr/bin/env python3
###

import sys,os,argparse,re,time
import logging
import pandas as pd

from ProteinGraphML.DataAdapter import OlegDB, TCRD
from ProteinGraphML.MLTools.StaticFeatures import staticData

###
if __name__ == "__main__":
  """
  This programs generates files for static features: lincs, hpa, gtex, and ccle. This code does not use training 
  or test data for the static features.
  """
  DBS=["olegdb", "tcrd"]
  SOURCES = ["gtex", "lincs", "ccle", "hpa"]
  parser = argparse.ArgumentParser(description="Generate static features for all proteins.")
  parser.add_argument("--db", required=True, choices=DBS, help="({0})".format("|".join(DBS)))

  parser.add_argument("--outputdir", default=".")
  parser.add_argument("--sources", help="comma-separated list: {0}".format(",".join(SOURCES)), default=(",".join(SOURCES)))
  parser.add_argument("--decimals", type=int, default=3) 
  parser.add_argument("-v", "--verbose", action="count", default=0)
  args = parser.parse_args()

  logging.basicConfig(format="%(levelname)s:%(message)s", level=(logging.DEBUG if args.verbose>1 else logging.INFO))

  if not args.sources:
    parser.error("--sources required.")

  sources = re.split("[, ]+", args.sources.strip())
  if len(set(sources) - set(SOURCES))>0:
    parser.error("Invalid sources: {0}".format(','.join(list(set(sources) - set(SOURCES)))))

  t0 = time.time()

  dbad = TCRD() if args.db == "tcrd" else OlegDB()

  if "gtex" in sources:
    ofile_gtex = args.outputdir+"/gtex.tsv"
    logging.info("GTEX: writing {0}".format(ofile_gtex))
    gtex = staticData.gtex(dbad)
    logging.info("GTEX: rows: {0}; cols: {1}".format(gtex.shape[0], gtex.shape[1]))
    gtex.round(args.decimals).to_csv(ofile_gtex, "\t", index=True)
    logging.info('{0}: elapsed time: {1}'.format(os.path.basename(sys.argv[0]), time.strftime('%Hh:%Mm:%Ss', time.gmtime(time.time()-t0))))

  if "hpa" in sources:
    ofile_hpa = args.outputdir+"/hpa.tsv"
    logging.info("HPA: writing {0}".format(ofile_hpa))
    hpa = staticData.hpa(dbad)
    logging.info("HPA: rows: {0}; cols: {1}".format(hpa.shape[0], hpa.shape[1]))
    hpa.round(args.decimals).to_csv(ofile_hpa, "\t", index=True)
    logging.info('{0}: elapsed time: {1}'.format(os.path.basename(sys.argv[0]), time.strftime('%Hh:%Mm:%Ss', time.gmtime(time.time()-t0))))

  if "lincs" in sources:
    ofile_lincs = args.outputdir+"/lincs.tsv"
    logging.info("LINCS: writing {0}".format(ofile_lincs))
    lincs = staticData.lincs(dbad)
    logging.info("LINCS: rows: {0}; cols: {1}".format(lincs.shape[0], lincs.shape[1]))
    lincs.round(args.decimals).to_csv(ofile_lincs, "\t", index=True)
    logging.info('{0}: elapsed time: {1}'.format(os.path.basename(sys.argv[0]), time.strftime('%Hh:%Mm:%Ss', time.gmtime(time.time()-t0))))

  if "ccle" in sources:
    try:
      ofile_ccle = args.outputdir+"/ccle.tsv"
      logging.info("CCLE: writing {0}".format(ofile_ccle))
      ccle = staticData.ccle(dbad)
      logging.info("CCLE: rows: {0}; cols: {1}".format(ccle.shape[0], ccle.shape[1]))
      ccle.round(args.decimals).to_csv(ofile_ccle, "\t", index=True)
    except Exception as e:
      logging.error("Failed to generate static features for CCLE: {0}".format(e))
    logging.info('{0}: elapsed time: {1}'.format(os.path.basename(sys.argv[0]), time.strftime('%Hh:%Mm:%Ss', time.gmtime(time.time()-t0))))

  logging.info('{0}: elapsed time: {1}'.format(os.path.basename(sys.argv[0]), time.strftime('%Hh:%Mm:%Ss', time.gmtime(time.time()-t0))))

