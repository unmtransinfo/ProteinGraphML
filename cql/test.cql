MATCH x=(n:GO)--(p1:Protein)--(p2:Protein)
WHERE n.Name CONTAINS "serotonin metabolic"
RETURN x LIMIT 50
;
