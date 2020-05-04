SELECT DISTINCT 
        COUNT(DISTINCT nhprotein_id) AS mouse_protein_ids, 
        COUNT(DISTINCT term_id) AS mp_term_ids,  
        COUNT(DISTINCT procedure_name) AS mp_procedure_names, 
        COUNT(DISTINCT parameter_name) AS mp_param_names, 
        gp_assoc AS association 
FROM
        phenotype 
WHERE
        ptype = 'IMPC'
GROUP BY
        association
;