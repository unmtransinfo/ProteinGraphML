SELECT DISTINCT
        nhprotein_id AS protein_id_m,
        term_id AS mp_term_id,
        term_name AS mp_term_name,
        p_value,
        effect_size,
        procedure_name,
        parameter_name,
        gp_assoc AS association
FROM
        phenotype
WHERE
        ptype = 'IMPC'
        ;

