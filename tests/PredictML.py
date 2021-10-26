#!/usr/bin/env python3
###
import sys, os, time, argparse, logging
import pickle

from proteingraphml.DataAdapter import OlegDB, selectAsDF, TCRD
from proteingraphml.MLTools.Data import BinaryLabel
from proteingraphml.MLTools.Procedures import *

"""
This program uses the machine learning model trained in program "TrainModelML.py" to predict the labels for the records
in the prediction data set. The prediction results are saved in a file.
"""
t0 = time.time()

PROCEDURES = ["XGBPredict"]
DBS = ["olegdb", "tcrd"]

parser = argparse.ArgumentParser(
    description="Run ML Procedure",
    epilog="--file must be specified; available procedures: {0}".format(
        str(PROCEDURES)
    ),
)
parser.add_argument(
    "procedure", choices=PROCEDURES, help="ML procedure to run"
)
parser.add_argument(
    "--predictfile",
    help='input file, pickled predict data, e.g. "diabetesPredictData.pkl"',
)
parser.add_argument("--modelfile", help="ML model file full path")
parser.add_argument(
    "--infofile", help="protein information file with full path"
)
parser.add_argument(
    "--resultdir",
    help='folder where results will be saved, e.g. "diabetes_no_lincs"',
)
parser.add_argument(
    "--db", choices=DBS, default="tcrd", help="{0}".format(str(DBS))
)
parser.add_argument(
    "-v", "--verbose", action="count", default=0, help="verbosity"
)

args = parser.parse_args()

logging.basicConfig(
    format="%(levelname)s:%(message)s",
    level=(logging.DEBUG if args.verbose > 1 else logging.INFO),
)

# get protein info file
if args.infofile is None:
    infofile = "data/plotDT.xlsx"
else:
    infofile = args.infofile

# get predict data from the file
if args.predictfile is None:
    parser.error("--predict data file must be specified.")
else:
    try:
        with open(args.predictfile, "rb") as f:
            predictData = pickle.load(f)
    except:
        logging.error(
            "Failed to open pickled predict data file {0}".format(
                args.predictfile
            )
        )
        exit()

# Get ML procedure
logging.info(
    "Procedure: {0} ({1})".format(args.procedure, locals()[args.procedure])
)

# directory and file name for the ML Model
if args.modelfile is None:
    logging.error("--modelfile required.")
    exit()
else:
    logging.info(
        "Model '{0}' will be used for prediction".format(args.modelfile)
    )

# Get result directory
if args.resultdir is not None:
    logging.info(
        "Results will be saved in directory: {0}".format(
            "results/" + args.resultdir
        )
    )
else:
    logging.error("Result directory is needed")
    exit()

# Access the db adaptor. Make TCRD as the default DB
dbAdapter = OlegDB() if args.db == "olegdb" else TCRD()

idDescription = dbAdapter.fetchPathwayIdDescription()  # fetch the description
idNameSymbol = (
    dbAdapter.fetchSymbolForProteinId()
)  # fetch name and symbol for protein

# call ML codes
d = BinaryLabel()
d.loadPredictData(predictData)

locals()[args.procedure](
    d, idDescription, idNameSymbol, args.modelfile, args.resultdir, infofile
)

logging.info(
    "{0}: elapsed time: {1}".format(
        os.path.basename(sys.argv[0]),
        time.strftime("%Hh:%Mm:%Ss", time.gmtime(time.time() - t0)),
    )
)
