SELECT DISTINCT
	clinvar.protein_id,
	p.sym,
	p.uniprot,
	p.description
FROM
	clinvar
JOIN
	protein p ON p.id = clinvar.protein_id
JOIN
	clinvar_phenotype ON clinvar.clinvar_phenotype_id = clinvar_phenotype.id
JOIN
	clinvar_phenotype_xref ON clinvar_phenotype.id = clinvar_phenotype_xref.clinvar_phenotype_id
WHERE 
	clinvar_phenotype_xref.source = 'OMIM'
	;
