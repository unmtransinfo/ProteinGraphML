#!/usr/bin/env python3
###

import sys, os, argparse, re, time
import logging

from proteingraphml.DataAdapter import OlegDB, TCRD
from proteingraphml.MLTools.StaticFeatures import staticData

###
if __name__ == "__main__":
    """
    This programs generates files for static features: lincs, hpa, gtex, and ccle. This code does not use training
    or test data for the static features.
    """
    DBS = ["olegdb", "tcrd"]
    SOURCES = ["gtex", "lincs", "ccle", "hpa"]
    parser = argparse.ArgumentParser(
        description="Generate static features for all proteins."
    )
    parser.add_argument(
        "--db", choices=DBS, default="tcrd", help=f"({'|'.join(DBS)})"
    )
    parser.add_argument("--outputdir", default=".")
    parser.add_argument(
        "--sources",
        help=f"comma-separated list: {','.join(SOURCES)}",
        default=(",".join(SOURCES)),
    )
    parser.add_argument("--decimals", type=int, default=3)
    parser.add_argument("-v", "--verbose", action="count", default=0)
    args = parser.parse_args()

    logging.basicConfig(
        format="%(levelname)s:%(message)s",
        level=(logging.DEBUG if args.verbose > 1 else logging.INFO),
    )

    if not args.sources:
        parser.error("--sources required.")

    sources = re.split("[, ]+", args.sources.strip())
    if len(set(sources) - set(SOURCES)) > 0:
        parser.error(
            f"Invalid sources: {','.join(list(set(sources) - set(SOURCES)))}"
        )

    t0 = time.time()

    # Make TCRD as the default DB
    dbad = OlegDB() if args.db == "olegdb" else TCRD()

    if "gtex" in sources:
        ofile_gtex = args.outputdir + "/gtex.tsv"
        logging.info(f"GTEX: writing {ofile_gtex}")
        gtex = staticData.gtex(dbad)
        logging.info(f"GTEX: rows: {gtex.shape[0]}; cols: {gtex.shape[1]}")
        gtex.round(args.decimals).to_csv(ofile_gtex, "\t", index=True)
        logging.info(
            "{0}: elapsed time: {1}".format(
                os.path.basename(sys.argv[0]),
                time.strftime("%Hh:%Mm:%Ss", time.gmtime(time.time() - t0)),
            )
        )

    if "hpa" in sources:
        ofile_hpa = args.outputdir + "/hpa.tsv"
        logging.info(f"HPA: writing {ofile_hpa}")
        hpa = staticData.hpa(dbad)
        logging.info(f"HPA: rows: {hpa.shape[0]}; cols: {hpa.shape[1]}")
        hpa.round(args.decimals).to_csv(ofile_hpa, "\t", index=True)
        logging.info(
            "{0}: elapsed time: {1}".format(
                os.path.basename(sys.argv[0]),
                time.strftime("%Hh:%Mm:%Ss", time.gmtime(time.time() - t0)),
            )
        )

    if "lincs" in sources:
        ofile_lincs = args.outputdir + "/lincs.tsv"
        logging.info(f"LINCS: writing {ofile_lincs}")
        lincs = staticData.lincs(dbad)
        logging.info(f"LINCS: rows: {lincs.shape[0]}; cols: {lincs.shape[1]}")
        lincs.round(args.decimals).to_csv(ofile_lincs, "\t", index=True)
        logging.info(
            "{0}: elapsed time: {1}".format(
                os.path.basename(sys.argv[0]),
                time.strftime("%Hh:%Mm:%Ss", time.gmtime(time.time() - t0)),
            )
        )

    if "ccle" in sources:
        try:
            ofile_ccle = args.outputdir + "/ccle.tsv"
            logging.info(f"CCLE: writing {ofile_ccle}")
            ccle = staticData.ccle(dbad)
            logging.info(f"CCLE: rows: {ccle.shape[0]}; cols: {ccle.shape[1]}")
            ccle.round(args.decimals).to_csv(ofile_ccle, "\t", index=True)
        except Exception as e:
            logging.error(f"Failed to generate static features for CCLE: {e}")
        logging.info(
            "{0}: elapsed time: {1}".format(
                os.path.basename(sys.argv[0]),
                time.strftime("%Hh:%Mm:%Ss", time.gmtime(time.time() - t0)),
            )
        )

    logging.info(
        "{0}: elapsed time: {1}".format(
            os.path.basename(sys.argv[0]),
            time.strftime("%Hh:%Mm:%Ss", time.gmtime(time.time() - t0)),
        )
    )
