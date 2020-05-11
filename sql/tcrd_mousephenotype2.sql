SELECT
        ph.nhprotein_id AS protein_id_m,
        ph.term_id AS mp_term_id,
        ph.term_name AS mp_term_name,
        ph.p_value,
        ph.effect_size,
        ph.procedure_name,
        ph.parameter_name,
        ph.gp_assoc AS association
FROM
        phenotype ph
WHERE
        ph.ptype = 'IMPC'
        ;

