MATCH
	x=(mp:MP {Name:"abnormal metabolism"})--(p1:Protein)--(g:GO)--(p2:Protein)
WITH p2, COUNT(x) AS score
RETURN p2.ID, p2.Name, score
ORDER BY score DESC
;
