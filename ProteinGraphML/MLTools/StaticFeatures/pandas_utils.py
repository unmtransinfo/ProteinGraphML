#!/usr/bin/env python3
###
import sys,os,argparse,re,pickle
import pandas

#############################################################################
def SearchRows(df, cols, coltags, qrys, rels, typs, fout):
  n = df.shape[0]
  for j,tag in enumerate(df.columns):
    if cols:
      if j not in cols: continue
      else: jj = cols.index(j)
    elif coltags:
      if tag not in coltags: continue
      else: jj = coltags.index(tag)
    #print("DEBUG: tag=%s; j=%d; jj=%d"%(tag,j,jj), file=sys.stderr)
    if qrys[jj].upper() in ('NA','NAN'):
      df = df[df[tag].isna()]
    elif typs[jj]=='int':
      df = df[df[tag].astype('int')==int(qrys[jj])]
    elif typs[jj]=='float':
      df = df[df[tag].astype('float')==float(qrys[jj])]
    else:
      df = df[df[tag].astype('str').str.match('^'+qrys[jj]+'$')]
  df.to_csv(fout, '\t', index=False)
  print("Rows found: %d / %d"%(df.shape[0],n), file=sys.stderr)

#############################################################################
if __name__=='__main__':
  parser = argparse.ArgumentParser(
        description='Pandas utilities for simple datafile transformations.')
  ops = ['csv2tsv', 'summary','showcols','selectcols','uvalcounts','colvalcounts','sortbycols','deduplicate','colstats','searchrows','pickle']
  compressions=['gzip', 'zip', 'bz2']
  parser.add_argument("op", choices=ops, help='operation')
  parser.add_argument("--i", dest="ifile", help="input (CSV|TSV)")
  parser.add_argument("--o", dest="ofile", help="output (CSV|TSV)")
  parser.add_argument("--coltags", help="cols specified by tag (comma-separated)")
  parser.add_argument("--cols", help="cols specified by idx (1+) (comma-separated)")
  parser.add_argument("--search_qrys", help="qrys (comma-separated, NA|NaN handled specially)")
  parser.add_argument("--search_rels", default="=", help="relationships (=|>|<) (comma-separated)")
  parser.add_argument("--search_typs", default="str", help="types (str|int|float) (comma-separated)")
  parser.add_argument("--compression", choices=compressions)
  parser.add_argument("--csv", action="store_true", help="delimiter is comma")
  parser.add_argument("--tsv", action="store_true", help="delimiter is tab")
  parser.add_argument("--disallow_bad_lines", action="store_true", help="default=allow+skip+warn")
  parser.add_argument("--nrows", type=int)
  parser.add_argument("--skiprows", type=int)
  parser.add_argument("-v", "--verbose", action="count")
  args = parser.parse_args()

  if args.op in ('selectcols', 'uvalcounts', 'colvalcounts', 'sortbycols'):
    if not (args.cols or args.coltags): 
      parser.error('%s requires --cols or --coltags.'%args.op)

  if not args.ifile:
    parser.error('Input file required.')

  if args.ofile:
    fout = open(args.ofile, "w")
  else:
    fout = sys.stdout

  if args.compression: compression=args.compression
  elif re.search('\.gz$', args.ifile, re.I): compression='gzip'
  elif re.search('\.bz2$', args.ifile, re.I): compression='bz2'
  elif re.search('\.zip$', args.ifile, re.I): compression='zip'
  else: compression=None

  if args.csv or args.op=='csv2tsv': delim=','
  elif args.tsv: delim='\t'
  elif re.search('\.csv', args.ifile, re.I): delim=','
  elif re.search('\.tsv', args.ifile, re.I) or re.search('\.tab', args.ifile, re.I): delim='\t'
  else: delim='\t'

  cols=None; coltags=None;
  if args.cols:
    cols = [(int(col.strip())-1) for col in re.split(r',', args.cols.strip())]
  elif args.coltags:
    coltags = [coltag.strip() for coltag in re.split(r',', args.coltags.strip())]

  search_qrys = [qry.strip() for qry in re.split(r',', args.search_qrys.strip())] if (args.search_qrys is not None) else None
  search_rels = [rel.strip() for rel in re.split(r',', args.search_rels.strip())] if (args.search_rels is not None) else None
  search_typs = [typ.strip() for typ in re.split(r',', args.search_typs.strip())] if (args.search_typs is not None) else None

  if args.op == 'showcols': args.nrows=1

  df = pandas.read_csv(args.ifile, sep=delim, compression=compression, error_bad_lines=args.disallow_bad_lines, nrows=args.nrows, skiprows=args.skiprows)

  if args.op == 'showcols':
    for j,tag in enumerate(df.columns):
      print('%d. "%s"'%(j+1,tag))

  elif args.op == 'summary':
    print("rows: %d ; cols: %d"%(df.shape[0], df.shape[1]))
    print("coltags: %s"%(', '.join(['"%s"'%tag for tag in df.columns])))

  elif args.op=='csv2tsv':
    df.to_csv(fout, '\t', index=False)

  elif args.op == 'selectcols':
    df = df[coltags] if coltags else df.iloc[:, cols]
    df.to_csv(fout, '\t', index=False)

  elif args.op == 'uvalcounts':
    for j,tag in enumerate(df.columns):
      if cols and j not in cols: continue
      if coltags and tag not in coltags: continue
      print('%d. %s: %d'%(j+1,tag,df[tag].nunique()))

  elif args.op == 'colvalcounts':
    for j,tag in enumerate(df.columns):
      if cols and j not in cols: continue
      if coltags and tag not in coltags: continue
      print('%d. %s:'%(j+1, tag))
      for key,val in df[tag].value_counts().iteritems():
        print('\t%6d: %s'%(val, key))

  elif args.op == 'colstats':
    for j,tag in enumerate(df.columns):
      if cols and j not in cols: continue
      if coltags and tag not in coltags: continue
      print('%d. %s:'%(j+1, tag))
      print('\tN: %d'%(df[tag].size))
      print('\tN_isna: %d'%(df[tag].isna().sum()))
      print('\tmin: %.2f'%(df[tag].min()))
      print('\tmax: %.2f'%(df[tag].max()))
      print('\tmean: %.2f'%(df[tag].mean()))
      print('\tmedian: %.2f'%(df[tag].median()))
      print('\tstd: %.2f'%(df[tag].std()))

  elif args.op == 'searchrows':
    if args.search_qrys is None: 
      parser.error('%s requires --search_qrys.'%args.op)
    #print("DEBUG: search_qrys=%s"%str(search_qrys), file=sys.stderr)
    #print("DEBUG: search_rels=%s"%str(search_rels), file=sys.stderr)
    #print("DEBUG: search_typs=%s"%str(search_typs), file=sys.stderr)
    SearchRows(df, cols, coltags, search_qrys, search_rels, search_typs, fout)

  elif args.op == 'pickle':
    if not args.ofile:
      parser.error('%s requires --o.'%args.op)
    fout.close()
    with open(args.ofile, 'wb') as fout:
      pickle.dump(df, fout, pickle.HIGHEST_PROTOCOL)

  else:
    parser.error('Unknown operation: %s'%args.op)
