SELECT DISTINCT
        ph.nhprotein_id AS protein_id_m,
        ph.term_id AS mp_term_id,
        mpo.name AS mp_term_name,
        ph.p_value,
        ph.effect_size,
        ph.procedure_name,
        ph.parameter_name,
        ph.gp_assoc AS association
FROM
        phenotype ph,
        mpo
WHERE
        ph.term_id = mpo.mpid
        AND
        ph.ptype = 'IMPC'
        ;
--        AND 
--        ph.term_id IN ('MP:0000180', 'MP:0005177', 'MP:0001549', 'MP:0005559')
--        ph.term_id IN ('MP:0000180', 'MP:0005177', 'MP:0001549')
--        ph.term_id = 'MP:0003947'
