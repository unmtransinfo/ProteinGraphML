import logging
import numpy as np

def basicPivot(df, key, column, value):
  return df.pivot_table(index=[key], columns=[column], values=value, aggfunc=np.median).fillna(0)

def gtex(DBadapter):
  df = DBadapter.loadGTEX()
  df = basicPivot(df, "protein_id", "tissue_type_detail", "median_tpm")
  logging.info("staticData: GTEX: rows: {0}; cols: {1}".format(df.shape[0], df.shape[1]))
  return df

def lincs(DBadapter):
  df = DBadapter.loadLINCS()
  df = basicPivot(df, "protein_id", "col_id", "zscore")
  logging.info("staticData: LINCS: rows: {0}; cols: {1}".format(df.shape[0], df.shape[1]))
  return df

#def ccle(DBadapter):
#  df = DBadapter.loadCCLE()

  #ccle[is.na(tissue), col_id := cell_id]
  #ccle[!is.na(tissue), col_id := sprintf("%s_%s", cell_id,tissue)]
  #ccle[, `:=`(tissue = NULL, cell_id = NULL)]

#  df = basicPivot(df, "protein_id", "col_id", "expression")
#  logging.info("staticData: CCLE: rows: {0}; cols: {1}".format(df.shape[0], df.shape[1]))
#  return df


#def hpa(DBadapter):
#  df = DBadapter.loadHPA()

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

#  df = basicPivot(df, "protein_id", "col_id", "level")
#  logging.info("staticData: HPA: rows: {0}; cols: {1}".format(df.shape[0], df.shape[1]))
#  return df


###
