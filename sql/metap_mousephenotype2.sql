SELECT 
        protein_id,
        mp_term_id,
        procedure_name,
        parameter_name,
        association
FROM 
        mousephenotype
WHERE
        mp_term_id = 'MP_0000180'
;