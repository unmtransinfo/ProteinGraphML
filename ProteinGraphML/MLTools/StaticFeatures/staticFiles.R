#!/usr/bin/env Rscript
###
library(data.table)
library(RPostgreSQL)
library(Matrix)

getmode <- function(v) {
  uniqv <- unique(v)
  uniqv[which.max(tabulate(match(v, uniqv)))]
}

replace_na <- function(DT, ind, replacement) {
  for(j in ind){
    set(DT, i = which(is.na(DT[[j]])), j = j, value = replacement)
  }
}

scale.data.table <- function(dt) {
  col.names <- colnames(dt)[2:ncol(dt)]
  dt[, (col.names) := lapply(.SD, scale), .SDcols=col.names]
}

conn <- dbConnect(PostgreSQL(), dbname = "metap",host='localhost',user='monty',pass='spam')


gtex <- dbGetQuery(conn, "select protein_id,median_tpm,tissue_type_detail from gtex")
setDT(gtex)
gtex <- dcast(gtex, protein_id ~ tissue_type_detail, fun.aggregate = median, value.var = "median_tpm", drop = T, fill = 0)
scale.data.table(gtex)
write.csv(gtex,file='gtex.csv')


ccle <- dbGetQuery(conn, "select protein_id,cell_id,tissue,expression from ccle")
setDT(ccle)
ccle[is.na(tissue), col_id := cell_id]
ccle[!is.na(tissue), col_id := sprintf("%s_%s", cell_id,tissue)]
ccle[, `:=`(tissue = NULL, cell_id = NULL)]
ccle <- dcast(ccle, protein_id ~ col_id, fun.aggregate = median, value.var = "expression", drop = T, fill = 0)
scale.data.table(ccle)
write.csv(ccle, file = "ccle.csv")


lincs <- dbGetQuery(conn, "select protein_id,pert_id||':'||cell_id as col_id,zscore from lincs")
setDT(lincs)
lincs <- dcast(lincs, protein_id ~ col_id, fun.aggregate = median, value.var = "zscore", drop = T, fill = 0)
scale.data.table(lincs)
write.csv(lincs,file='lincs.csv')


hpa <- dbGetQuery(conn, "select protein_id, tissue||'.'||cell_type as col_id,level from hpa_norm_tissue where reliability in ('supported','approved')")
setDT(hpa)
hpa$level <- factor(hpa$level, levels = c("not detected", "low", "medium", "high"), ordered = F)
hpa <- unique(hpa)
hpa[, col_id := gsub(" ", "_", col_id, fixed = T)]
hpa[, col_id := gsub("/", "_", col_id, fixed = T)]
hpa[, col_id := gsub(",", "", col_id, fixed = T)]
hpa[, col_id := gsub("-", "_", col_id, fixed = T)]
hpa <- dcast(hpa, protein_id ~ col_id, fun.aggregate = getmode, value.var = "level", drop = T, fill = "not detected")
replace_na(hpa, 2:ncol(hpa), "not detected")
hpa.sparse.matrix <- sparse.model.matrix(~.-1, data = hpa)
hpa <- as.data.table(as.matrix(hpa.sparse.matrix), keep.rownames = F)
write.csv(hpa,file='hpa.csv')

