MATCH (n:GO) WHERE SUBSTRING(n.Name, 0, 2) = 'P:' SET n.GO_Type = 'Process' ;
MATCH (n:GO) WHERE SUBSTRING(n.Name, 0, 2) = 'F:' SET n.GO_Type = 'Function' ;
MATCH (n:GO) WHERE SUBSTRING(n.Name, 0, 2) = 'C:' SET n.GO_Type = 'Compartment' ;
