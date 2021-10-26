import logging
import pandas as pd
import numpy as np


def basicPivot(df, key, column, value):
    return df.pivot_table(
        index=[key], columns=[column], values=value, aggfunc=np.median
    ).fillna(0)


def gtex(dbad):
    df = dbad.loadGTEX()
    logging.info(
        "staticData: DBAdapter:{0}; GTEX: rows: {1}; cols: {2}".format(
            type(dbad).__name__, df.shape[0], df.shape[1]
        )
    )
    df.info()  # DEBUG
    df = basicPivot(df, "protein_id", "tissue_type_detail", "median_tpm")
    df.reset_index(drop=False, inplace=True)
    logging.info(
        "staticData: GTEX proteins: rows: {0}; cols: {1}".format(
            df.shape[0], df.shape[1]
        )
    )
    return df


def lincs(dbad):
    df = dbad.loadLINCS()
    logging.info(
        "staticData: DBAdapter:{0}; LINCS: rows: {1}; cols: {2}".format(
            type(dbad).__name__, df.shape[0], df.shape[1]
        )
    )
    df = basicPivot(df, "protein_id", "col_id", "zscore")
    df.reset_index(drop=False, inplace=True)
    logging.info(
        "staticData: LINCS proteins: rows: {0}; cols: {1}".format(
            df.shape[0], df.shape[1]
        )
    )
    return df


def ccle(dbad):
    df = dbad.loadCCLE()
    logging.info(
        "staticData: DBAdapter:{0}; CCLE: rows: {1}; cols: {2}".format(
            type(dbad).__name__, df.shape[0], df.shape[1]
        )
    )
    df["col_id"] = df.cell_id + "_" + df.tissue
    df.col_id = df.col_id.str.replace("[ /,]", "_")
    df = df[["protein_id", "col_id", "expression"]].drop_duplicates()
    df = basicPivot(df, "protein_id", "col_id", "expression")
    df.reset_index(drop=False, inplace=True)
    logging.info(
        "staticData: CCLE proteins: rows: {0}; cols: {1}".format(
            df.shape[0], df.shape[1]
        )
    )
    return df


def hpa(dbad):
    # (Why did Oleg use mode not median?)
    df = dbad.loadHPA()
    logging.debug(
        "staticData ({0}): HPA: rows: {1}; cols: {2}".format(
            type(dbad).__name__, df.shape[0], df.shape[1]
        )
    )
    df = df.drop_duplicates()
    df.col_id = df.col_id.str.replace("[ /,]", "_")
    df = df.rename(columns={"level": "level_str"})
    for key, val in df["level_str"].value_counts().iteritems():
        logging.debug("\t%s: %6d: %s" % ("level_str", val, key))
    df["level"] = df.level_str.apply(
        lambda s: 3
        if s == "High"
        else 2
        if s == "Medium"
        else 1
        if s == "Low"
        else 0
        if "Not detected"
        else 0
    )
    for key, val in df["level"].value_counts().iteritems():
        logging.debug("\t%s: %6d: %s" % ("level", val, key))
    logging.debug(
        "staticData ({0}): HPA: rows: {1}; cols: {2}".format(
            type(dbad).__name__, df.shape[0], df.shape[1]
        )
    )
    # df.info() #DEBUG
    df = basicPivot(df, "protein_id", "col_id", "level")
    logging.debug(
        "staticData ({0}): HPA: rows: {1}; cols: {2}".format(
            type(dbad).__name__, df.shape[0], df.shape[1]
        )
    )
    df.reset_index(drop=False, inplace=True)
    return df


###
