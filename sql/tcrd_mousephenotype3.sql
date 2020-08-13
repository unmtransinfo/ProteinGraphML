SELECT DISTINCT pr.id AS protein_id_h,
    ph.nhprotein_id AS protein_id_m,
    pr.sym AS human_sym,
    nhpr.sym AS mouse_sym,
    ph.term_id AS mp_term_id,
    ph.term_name AS mp_term_name,
    ph.p_value,
    ph.procedure_name,
    ph.gp_assoc AS association
FROM protein pr
JOIN ortholog orth ON pr.id = orth.protein_id
JOIN nhprotein nhpr ON nhpr.sym = orth.symbol
JOIN phenotype ph ON ph.nhprotein_id = nhpr.id
WHERE orth.taxid = 10090 AND
      nhpr.taxid = 10090 AND
      ptype = 'IMPC' AND
      ph.term_id ='MP:0005559'
ORDER BY pr.sym ;