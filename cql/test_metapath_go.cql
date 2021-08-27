MATCH
	x=(mp:MP {Name:"retinal degeneration"})--(p1:Protein)--(n:GO)--(p2:Protein)
WHERE
	p2.Name CONTAINS "Cyclic AMP-dependent transcription factor ATF-4"
RETURN x
;
