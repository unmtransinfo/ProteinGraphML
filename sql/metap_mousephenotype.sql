SELECT 
        COUNT(DISTINCT protein_id) AS protein_ids,
        COUNT(DISTINCT mp_term_id) AS mp_term_ids,
        COUNT(DISTINCT procedure_name) AS mp_procedure_names,
        COUNT(DISTINCT parameter_name) AS mp_param_names,
        association
FROM 
        mousephenotype
GROUP BY
        association
;