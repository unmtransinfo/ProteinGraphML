#!/usr/bin/env python3
"""
Analyze ProteinGraphML training data, from input pickle file.
E.g. diabetes_pid_TrainingData.pkl

Example output:
 ./pickle_utils.py diabetes_pid_TrainingData.pkl 
INFO:rows: 2105; columns: 22056
INFO:Y:   1975: 0
INFO:Y:    130: 1
INFO:PPI columns : 0/22057 (0.00%)
INFO:cell line columns : 18997/22057 (86.13%)
INFO:GO columns : 785/22057 (3.56%)
INFO:IPR columns : 362/22057 (1.64%)
INFO:KEGG/R-HSA columns : 340/22057 (1.54%)
INFO:KEGG/hsa columns : 117/22057 (0.53%)
INFO:ACH columns : 1156/22057 (5.24%)
INFO:0.     6: features: 2868 / 22054 ; missing 0 / 22054; zeros 19186 / 22054
INFO:1. 16903: features: 1242 / 22054 ; missing 0 / 22054; zeros 20812 / 22054
INFO:2.  9740: features: 2500 / 22054 ; missing 0 / 22054; zeros 19554 / 22054
INFO:3.  3599: features: 21674 / 22054 ; missing 0 / 22054; zeros 380 / 22054
INFO:4. 15387: features: 2933 / 22054 ; missing 0 / 22054; zeros 19121 / 22054

"""
import sys,os,pickle,logging
import pandas as pd


if __name__=="__main__":

  logging.basicConfig(format='%(levelname)s:%(message)s', level=(logging.DEBUG))

  if len(sys.argv)<2:
    logging.error(f"Syntax: {sys.argv[0]} PICKLEFILE")
    
  with open(sys.argv[1], 'rb') as f:
    df = pickle.load(f)

  logging.info(f"rows: {df.shape[0]}; columns: {df.shape[1]}")

  df.reset_index(level=0, inplace=True) #protein_id

  tag = "Y"
  for key,val in df[tag].value_counts().iteritems():
    logging.info(f"{tag}: {val:6d}: {key}")

  #for tag in df.columns: print(f"{tag}")
  coltags = pd.Series([tag for tag in df.columns])

  # Columns with PIDs:
  ppi_match = coltags.str.fullmatch('[0-9]+\s*')
  logging.info(f"PPI columns : {ppi_match.sum()}/{df.shape[1]} ({100*ppi_match.sum()/df.shape[1]:.2f}%)")

  # Columns with cell lines? co-expression?:
  cell_match = coltags.str.match('[0-9]+:[A-Z]')
  logging.info(f"cell line columns : {cell_match.sum()}/{df.shape[1]} ({100*cell_match.sum()/df.shape[1]:.2f}%)")

  # Columns with GO:
  go_match = coltags.str.match('GO:')
  logging.info(f"GO columns : {go_match.sum()}/{df.shape[1]} ({100*go_match.sum()/df.shape[1]:.2f}%)")

  # Columns with IPR:
  ipr_match = coltags.str.match('IPR')
  logging.info(f"IPR columns : {ipr_match.sum()}/{df.shape[1]} ({100*ipr_match.sum()/df.shape[1]:.2f}%)")

  # Columns with HSA (KEGG):
  rhsa_match = coltags.str.match('R-HSA')
  logging.info(f"KEGG/R-HSA columns : {rhsa_match.sum()}/{df.shape[1]} ({100*rhsa_match.sum()/df.shape[1]:.2f}%)")
  # Columns with hsa (KEGG):
  hsa_match = coltags.str.match('hsa')
  logging.info(f"KEGG/hsa columns : {hsa_match.sum()}/{df.shape[1]} ({100*hsa_match.sum()/df.shape[1]:.2f}%)")

  # Columns with ACH:
  ach_match = coltags.str.match(r'.*\(ACH-')
  logging.info(f"ACH columns : {ach_match.sum()}/{df.shape[1]} ({100*ach_match.sum()/df.shape[1]:.2f}%)")

  ###
  # Training positives
  # Counts and categories of non-empty features.

  for i in df[df["Y"]==1].index:
    protein_id = int(df.iloc[i,]["protein_id"])
    fvec = df.iloc[i,3:]
    missing = (fvec.isna() | fvec.isnull())
    zeros = fvec==0
    logging.info(f"{i}. {protein_id:5d}: features: {fvec.size-missing.sum()-zeros.sum()} / {fvec.size} ; missing {missing.sum()} / {fvec.size}; zeros {zeros.sum()} / {fvec.size}")
    
