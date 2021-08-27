MATCH
        x=(mp:MP)--(p1:Protein)--(i:INTERPRO)--(p2:Protein)
WHERE 
        mp.Name = "increased circulating LDL cholesterol level"
        AND p2.Name CONTAINS "ligase"
RETURN x
;
