MATCH
	x=(mp:MP)--(p1:Protein)--(n:GO)--(p2:Protein)
WHERE
	mp.Name = "retinal degeneration"
	AND p2.Name CONTAINS "Cyclic AMP-dependent transcription factor ATF-4"
RETURN x
;
