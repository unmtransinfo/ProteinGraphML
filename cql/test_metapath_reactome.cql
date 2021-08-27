MATCH
        x=(mp:MP {Name:"unresponsive to tactile stimuli"})--(p1:Protein)--(r:REACTOME)--(p2:Protein)
WHERE 
        r.Name =~ ".*[mM]etabolism.*"
        AND p2.Name CONTAINS "aminotransferase"
RETURN x
;
