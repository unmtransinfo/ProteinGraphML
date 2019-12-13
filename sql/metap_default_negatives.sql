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
        AND clinvar.clinical_significance IN ('Pathogenic, Affects','Benign, protective, risk factor','Pathogenic/Likely pathogenic','Pathogenic/Likely pathogenic, other','Pathogenic, other','Affects','Pathogenic, other,
protective','Conflicting interpretations of pathogenicity, Affects, association, other','Pathogenic/Likely pathogenic, drug response','Pathogenic, risk factor','risk factor','Pathogenic, association','Conflicting interpretations of pathogenicity, Affects,
association, risk factor','Pathogenic/Likely pathogenic, risk factor','Affects, risk factor','Conflicting interpretations of pathogenicity, association, other, risk factor','Likely pathogenic, association','association, protective','Likely pathogenic,
Affects','Pathogenic','Conflicting interpretations of pathogenicity, association','Pathogenic/Likely pathogenic, Affects, risk factor','Conflicting interpretations of pathogenicity, other, risk factor','association, risk factor','Benign, protective','Conflicting
interpretations of pathogenicity, risk factor','Uncertain significance, protective','association','Uncertain significance, Affects','protective,
risk factor','Pathogenic, association, protective','Pathogenic, protective','Likely pathogenic, other','Pathogenic, protective, risk factor','Benign, association, protective','Conflicting interpretations of pathogenicity, Affects','Benign/Likely benign, protective','protective')
;
