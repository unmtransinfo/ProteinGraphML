import logging
import pandas as pd
import numpy as np

def basicPivot(df, key, column, value):
  return df.pivot_table(index=[key], columns=[column], values=value, aggfunc=np.median).fillna(0)

def gtex(DBadapter):
  df = DBadapter.loadGTEX()
  logging.info("staticData: GTEX DBadapter: rows: {0}; cols: {1}".format(df.shape[0], df.shape[1]))
  df = basicPivot(df, "protein_id", "tissue_type_detail", "median_tpm")
  df.reset_index(drop=False, inplace=True)
  logging.info("staticData: GTEX proteins: rows: {0}; cols: {1}".format(df.shape[0], df.shape[1]))
  return df

def lincs(DBadapter):
  df = DBadapter.loadLINCS()
  logging.info("staticData: LINCS DBadapter: rows: {0}; cols: {1}".format(df.shape[0], df.shape[1]))
  df = basicPivot(df, "protein_id", "col_id", "zscore")
  df.reset_index(drop=False, inplace=True)
  logging.info("staticData: LINCS proteins: rows: {0}; cols: {1}".format(df.shape[0], df.shape[1]))
  return df

def ccle(DBadapter):
  df = DBadapter.loadCCLE()
  logging.info("staticData: CCLE DBadapter: rows: {0}; cols: {1}".format(df.shape[0], df.shape[1]))
  df["col_id"] = df.cell_id
  df.loc[df.tissue.notna(), "col_id"] = (df.cell_id+"_"+df.tissue)
  df = df[["protein_id", "col_id", "expression"]].drop_duplicates()
  df = basicPivot(df, "protein_id", "col_id", "expression")
  df.reset_index(drop=False, inplace=True)
  logging.info("staticData: CCLE proteins: rows: {0}; cols: {1}".format(df.shape[0], df.shape[1]))
  return df

def hpa(DBadapter):
  df = DBadapter.loadHPA()
  logging.info("staticData: HPA DBadapter: rows: {0}; cols: {1}".format(df.shape[0], df.shape[1]))

  #R:
  #(Why did Oleg use mode not median?)
  #hpa$level <- factor(hpa$level, levels = c("not detected", "low", "medium", "high"), ordered=F)
  #hpa <- unique(hpa)
  #hpa[, col_id := gsub(" ", "_", col_id, fixed = T)]
  #hpa[, col_id := gsub("/", "_", col_id, fixed = T)]
  #hpa[, col_id := gsub(",", "", col_id, fixed = T)]
  #hpa[, col_id := gsub("-", "_", col_id, fixed = T)]
  #hpa <- dcast(hpa, protein_id ~ col_id, fun.aggregate = getmode, value.var = "level", drop = T, fill = "not detected")
  #replace_na(hpa, 2:ncol(hpa), "not detected")
  #hpa.sparse.matrix <- sparse.model.matrix(~.-1, data = hpa)
  #hpa <- as.data.table(as.matrix(hpa.sparse.matrix), keep.rownames = F)

  df = df.drop_duplicates()
  df.col_id = df.col_id.str.replace(",", "", regex=False)
  df.col_id = df.col_id.str.replace("[ /-]", "_")
  df = df.rename(columns={'level':'level_str'})
  df["level"] = pd.Series(dtype=int)
  df.loc[df.level_str == "not detected", "level"] = 0
  df.loc[df.level_str == "low", "level"] = 1
  df.loc[df.level_str == "medium", "level"] = 2
  df.loc[df.level_str == "high", "level"] = 3
  df.loc[df.level_str.isna(), "level"] = 0
  df = basicPivot(df, "protein_id", "col_id", "level")
  df.fillna(0)
  df.reset_index(drop=False, inplace=True)
  logging.info("staticData: HPA proteins: rows: {0}; cols: {1}".format(df.shape[0], df.shape[1]))
  return df

###
