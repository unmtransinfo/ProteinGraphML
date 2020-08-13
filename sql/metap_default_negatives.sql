--
-- TSV: psql -qAF $'\t' 
--
SELECT DISTINCT
        clinvar.protein_id,
        protein.accession,
        protein.symbol,
        protein.name
--        clinvar.clinical_significance,
--        clinvar_disease.cv_dis_id,
--        clinvar_disease.phenotype,
--        clinvar_disease_xref.source,
--        clinvar_disease_xref.source_id
FROM
        clinvar_disease,
        clinvar_disease_xref,
        clinvar
JOIN
        protein ON protein.protein_id = clinvar.protein_id
WHERE
        clinvar_disease.cv_dis_id=clinvar_disease_xref.cv_dis_id 
        AND clinvar_disease_xref.source = 'OMIM' 
        AND clinvar.cv_dis_id=clinvar_disease.cv_dis_id 
        AND clinvar.clinical_significance IN (
	'Affects',
	'Affects, risk factor',
	'Benign, association, protective',
	'Benign, protective',
	'Benign, protective, risk factor',
	'Benign/Likely benign, protective',
	'Conflicting interpretations of pathogenicity, Affects',
	'Conflicting interpretations of pathogenicity, Affects, association, other',
	'Conflicting interpretations of pathogenicity, Affects, association, risk factor',
	'Conflicting interpretations of pathogenicity, association',
	'Conflicting interpretations of pathogenicity, association, other, risk factor',
	'Conflicting interpretations of pathogenicity, other, risk factor',
	'Conflicting interpretations of pathogenicity, risk factor',
	'Likely pathogenic, Affects',
	'Likely pathogenic, association',
	'Likely pathogenic, other',
	'Pathogenic',
	'Pathogenic, Affects',
	'Pathogenic, association',
	'Pathogenic, association, protective',
	'Pathogenic, other',
	'Pathogenic, other, protective',
	'Pathogenic, protective',
	'Pathogenic, protective, risk factor',
	'Pathogenic, risk factor',
	'Pathogenic/Likely pathogenic',
	'Pathogenic/Likely pathogenic, Affects, risk factor',
	'Pathogenic/Likely pathogenic, drug response',
	'Pathogenic/Likely pathogenic, other',
	'Pathogenic/Likely pathogenic, risk factor',
	'Uncertain significance, Affects',
	'Uncertain significance, protective',
	'association',
	'association, protective',
	'association, risk factor',
	'protective',
	'protective, risk factor',
	'risk factor'
	)
;
