
import numpy as np

def gtex(DBadapter):
	# each of these loads the data
	#gtex <- dbGetQuery(conn, "select protein_id,median_tpm,tissue_type_detail from gtex")
	#setDT(gtex)
	#gtex <- dcast(gtex, protein_id ~ tissue_type_detail, fun.aggregate = median, value.var = "median_tpm", drop = T, fill = 0)
	return basicPivot(DBadapter.loadGTEX(),"protein_id","tissue_type_detail","median_tpm")


def ccle(DBadapter):
	return basicPivot(DBadapter.loadGTEX(),"protein_id","tissue_type_detail","median_tpm")


# THESE FEATURES ARE COMING ONLINE

#def ccle(DBadapter):

#def lincs(DBadapter):

#def hpa(DBadapter):


def basicPivot(data,key,column,value):
	return data.pivot_table(index=[key],columns=[column], values=value,aggfunc=np.median).fillna(0)

	