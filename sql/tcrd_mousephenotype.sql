-- nh = non-human
SELECT COUNT(DISTINCT 
        nhprotein_id, 
        term_id, 
        term_name, 
        p_value, 
        effect_size, 
        procedure_name, 
        parameter_name, 
        gp_assoc
        ) AS IMPC_CT 
FROM
        phenotype
WHERE
        ptype = 'IMPC'        
;