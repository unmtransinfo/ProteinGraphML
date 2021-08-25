MATCH
        x=(mp:MP)--(p1:Protein)--(k:KEGG)--(p2:Protein)
WHERE 
        mp.Name = "increased circulating LDL cholesterol level"
        AND p2.Name CONTAINS "Transcription factor AP-1"
        AND k.Name = "cAMP signaling pathway"
RETURN x
;
