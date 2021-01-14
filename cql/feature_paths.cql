MATCH x=(n:REACTOME)-[]-(p:Protein) WHERE n.Name = 'PI Metabolism' RETURN x  LIMIT 25 ;
MATCH x=(p1:Protein)-[r:STRING]-(p2:Protein) WHERE p1.ID = '19402' RETURN x LIMIT 25 ;
MATCH x=(g:GO)-[]-(p:Protein) WHERE g.Name CONTAINS 'glutamatergic synapse' RETURN x LIMIT 25 ;
