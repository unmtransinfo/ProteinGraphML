MATCH paths = (mp:MP {Name:"abnormal metabolism"})--(p1:Protein)--(r:REACTOME)--(p:Protein)
WITH mp,p,paths LIMIT 25
UNWIND paths AS pathz
UNWIND pathz AS path
WITH mp,p,(NODES(path)) AS pnodes, REDUCE(tot = 0.0, mnode IN NODES(path) | tot + SIZE((mnode)-[]-()) ^-.4) AS dwp
RETURN mp.ID, mp.Name, p.Name, SIZE(pnodes) AS numberOfNodes, SUM(dwp) AS DWPC
ORDER BY DWPC DESC ;
