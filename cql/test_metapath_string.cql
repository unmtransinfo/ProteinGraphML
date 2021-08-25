MATCH
        x=(mp:MP {Name:"increased circulating LDL cholesterol level"})--(p1:Protein)--(p2:Protein)
WHERE 
        p1.Name =~ ".*receptor.*"
RETURN x
;
