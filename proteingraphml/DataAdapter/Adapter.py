import os
from pony.orm import *
import pandas as pd
import yaml
import logging


from .biodata_helper import selectAsDF, attachColumn, generateDepthMap


# our graph takes a list of pandas frames, w/ relationships, and constructs a graph from all of it ... we may wrap
# that, but the adapter should provide pandas frames


class GraphEdge:
    # nodeLeft =nodeRight,association = None
    directed = False

    def __init__(self, nodeLeft, nodeRight, edge=None, data=None):
        self.nodeLeft = nodeLeft
        self.nodeRight = nodeRight

        if edge is None:
            self.edge = None
        else:
            self.edge = edge

        self.data = data

    def setDirected(self):
        self.directed = True


class NodeName:
    # this will store, a key type, and a name
    # when we make one of these, we can call it later in the graph to rejoin with the nodes in question

    keyValue = None  # the value we can construct
    name = None  # so we can call .name("Protein",nodes)
    dataframe = None  # a mapping we can use

    def __init__(self, name, keyValue, dataframe):
        self.keyValue = keyValue
        self.name = name
        self.dataframe = dataframe


class Adapter:
    # when we produce, we'd call a mat mult of each of the metapath chunks... stitch them together at the end
    # we'd use the associations known from the DB..

    # the adapater will have to add new kinds of edges, so you'd want to include an edge adder, adds a set of edges

    graph = None
    names = []

    def attachEdges(self):
        # maybe pass in an edge object here? metapath type? an intermediate object, which has relationships, can stack
        pass

    def makeBaseGraph(self):
        # edges = [self.geneToDisease,self.mouseToHumanAssociation,self.PPI]
        edges = [
            self.mouseToHumanAssociation,
            self.geneToDisease,
        ]  # self.PPI] # order matters!!
        graph = self.graphBuilder(edges)

        # adding string DB, would attach another graph, and save a separate one for us ...
        self.graph = graph
        return graph

    def saveNameMap(
        self, label, key, name, frame
    ):  # this will drop all columns except for key/name

        # NodeName("MP","mp_term_id",mpOnto.drop(['parent_id'],axis=1).drop_duplicates())

        newNode = NodeName(label, key, frame[[key, name]].drop_duplicates())
        self.names.append(newNode)


class OlegDB(Adapter):
    config_file = os.environ["HOME"] + "/.ProteinGraphML.yaml"
    # config_file = "/code/DBcreds.yaml"

    GTD = None
    mouseToHumanAssociation = None
    geneToDisease = None
    childParentDict = None

    db = None

    def __init__(self):
        self.load()

    def loadTotalProteinList(self):

        protein = selectAsDF(
            "select protein_id from protein where tax_id = 9606",
            ["protein_id"],
            self.db,
        )
        logging.debug(
            "(OlegDB.loadTotalProteinList) Human protein IDs returned: {0}".format(
                protein.shape[0]
            )
        )

        return protein

    def loadReactome(self, proteinFilter=None):

        reactome = selectAsDF(
            "select * from reactomea",
            ["protein_id", "reactome_id", "evidence"],
            self.db,
        )

        if proteinFilter is not None:
            reactome = reactome[reactome["protein_id"].isin(proteinFilter)]

        logging.debug(
            "(OlegDB.loadReactome) Reactome rows returned: {0}".format(
                reactome.shape[0]
            )
        )
        return GraphEdge("protein_id", "reactome_id", data=reactome)

    def loadPPI(self, proteinFilter=None):

        stringDB = selectAsDF(
            "select * from stringdb_score",
            ["protein_id1", "protein_id2", "combined_score"],
            self.db,
        )

        if proteinFilter is not None:
            stringDB = stringDB[stringDB["protein_id1"].isin(proteinFilter)]
            stringDB = stringDB[stringDB["protein_id2"].isin(proteinFilter)]

        logging.debug(
            "(OlegDB.loadPPI) STRING rows returned: {0}".format(
                stringDB.shape[0]
            )
        )
        return GraphEdge(
            "protein_id1", "protein_id2", "combined_score", stringDB
        )

    def loadKegg(self, proteinFilter=None):
        kegg = selectAsDF(
            "select protein_id,kegg_pathway_id from kegg_pathway",
            ["protein_id", "kegg_pathway_id"],
            self.db,
        )

        if proteinFilter is not None:
            kegg = kegg[kegg["protein_id"].isin(proteinFilter)]

        logging.debug(
            "(OlegDB.loadKegg) KEGG rows returned: {0}".format(kegg.shape[0])
        )
        return GraphEdge("protein_id", "kegg_pathway_id", data=kegg)

    def loadInterpro(self, proteinFilter=None):

        interpro = selectAsDF(
            "select distinct protein_id,entry_ac from interproa",
            ["protein_id", "entry_ac"],
            self.db,
        )

        if proteinFilter is not None:
            interpro = interpro[interpro["protein_id"].isin(proteinFilter)]

        logging.debug(
            "(OlegDB.loadInterpro) Interpro rows returned: {0}".format(
                interpro.shape[0]
            )
        )
        return GraphEdge("protein_id", "entry_ac", data=interpro)

    def loadGo(self, proteinFilter=None):

        goa = selectAsDF(
            "select protein_id,go_id from goa",
            ["protein_id", "go_id"],
            self.db,
        )

        if proteinFilter is not None:
            goa = goa[goa["protein_id"].isin(proteinFilter)]

        logging.debug(
            "(OlegDB.loadGo) GO rows returned: {0}".format(goa.shape[0])
        )
        return GraphEdge("protein_id", "go_id", data=goa)

    # static features
    def loadGTEX(self):
        gtex = selectAsDF(
            "SELECT protein_id, median_tpm, tissue_type_detail FROM gtex",
            ["protein_id", "median_tpm", "tissue_type_detail"],
            self.db,
        )
        logging.debug(
            "(OlegDB.loadGTEX) GTEx rows returned: {0}".format(gtex.shape[0])
        )
        return gtex

    def loadCCLE(self):
        ccle = selectAsDF(
            "SELECT protein_id, cell_id, tissue, expression FROM ccle",
            ["protein_id", "cell_id", "tissue", "expression"],
            self.db,
        )
        logging.debug(
            "(OlegDB.loadCCLE) CCLE rows returned: {0}".format(ccle.shape[0])
        )
        return ccle

    def loadLINCS(self):
        lincs = selectAsDF(
            "SELECT protein_id, pert_id||':'||cell_id AS col_id, zscore FROM lincs",
            ["protein_id", "col_id", "zscore"],
            self.db,
        )
        logging.debug(
            "(OlegDB.loadLINCS) LINCS rows returned: {0}".format(
                lincs.shape[0]
            )
        )
        return lincs

    def loadHPA(self):
        hpa = selectAsDF(
            "SELECT protein_id, tissue||'.'||cell_type AS col_id, level FROM hpa_norm_tissue WHERE reliability IN ("
            "'supported','approved')",
            ["protein_id", "col_id", "level"],
            self.db,
        )
        logging.debug(
            "(OlegDB.loadHPA) HPA rows returned: {0}".format(hpa.shape[0])
        )
        return hpa

    #
    def load(self):
        with open(self.config_file, "r") as stream:
            try:
                credentials = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                logging.error(
                    "DB credentials not found in {0}: {1}".format(
                        self.config_file, str(exc)
                    )
                )

        user = credentials["user"]
        password = credentials["password"]
        host = credentials["host"]
        database = credentials["database"]

        self.db = Database()
        self.db.bind(
            provider="postgres",
            user=user,
            password=password,
            host=host,
            database=database,
        )
        logging.debug(
            "(OlegDB.load) Connected to db (%s): %s:%s:%s"
            % (self.db.provider_name, host, database, user)
        )

        self.db.generate_mapping(create_tables=False)

        # hack ... saving the (DB) like this
        db = self.db
        # select everything from the DB
        TableColumns = ["hid", "homologene_group_id", "tax_id", "protein_id"]

        humanProteinList = selectAsDF(
            "SELECT * FROM homology WHERE tax_id = 9606", TableColumns, db
        )
        logging.debug(
            "(OlegDB.load) humanProteinList: %d" % (humanProteinList.shape[0])
        )
        mouseProteinList = selectAsDF(
            "select * from homology WHERE tax_id = 10090", TableColumns, db
        )
        logging.debug(
            "(OlegDB.load) mouseProteinList: %d" % (mouseProteinList.shape[0])
        )
        mousePhenotype = selectAsDF(
            "select * from mousephenotype",
            [
                "protein_id",
                "mp_term_id",
                "p_value",
                "effect_size",
                "procedure_name",
                "parameter_name",
                "association",
            ],
            db,
        )
        logging.debug(
            "(OlegDB.load) mousePhenotype: %d" % (mousePhenotype.shape[0])
        )
        mpOnto = selectAsDF(
            "select * from mp_onto", ["mp_term_id", "parent_id", "name"], db
        )
        logging.debug("(OlegDB.load) mpOnto: %d" % (mpOnto.shape[0]))

        self.saveNameMap(
            "MP_ontology", "mp_term_id", "name", mpOnto
        )  # we will save this data to the graph, so we can get it later

        mouseToHumanMap = self.buildHomologyMap(
            humanProteinList, mouseProteinList
        )
        combinedSet = attachColumn(
            mouseToHumanMap, mousePhenotype, "protein_id"
        )  # just bind the protein ID from our last table
        mouseToHumanAssociation = combinedSet[
            ["protein_id_h", "mp_term_id", "association"]
        ].drop_duplicates()
        logging.debug(
            "(OlegDB.load) mouseToHumanAssociation: %d"
            % (mouseToHumanAssociation.shape[0])
        )

        def getVal(row):
            return depthMap[row["mp_term_id"]]

        depthMap = generateDepthMap(mpOnto)
        mpOnto["level"] = mpOnto.apply(getVal, axis=1)
        mpOnto = mpOnto[mpOnto["level"] > 1]  # remove the single level stuff
        geneToDisease = attachColumn(
            mouseToHumanAssociation, mpOnto, "mp_term_id"
        )

        # we could extract this piece layer
        self.geneToDisease = GraphEdge(
            "mp_term_id",
            "protein_id_h",
            edge="association",
            data=mouseToHumanAssociation,
        )
        parentHierarchy = GraphEdge(
            "mp_term_id",
            "parent_id",
            edge=None,
            data=geneToDisease[["mp_term_id", "parent_id"]],
        )
        parentHierarchy.setDirected()
        self.phenotypeHierarchy = parentHierarchy

        # this child dict saves parents in reverse order, so that you can look them up directly
        childParentDict = {}
        for fval, val in zip(
            self.phenotypeHierarchy.data["mp_term_id"],
            self.phenotypeHierarchy.data["parent_id"],
        ):
            if fval not in childParentDict.keys():
                childParentDict[fval] = set([val])
            else:
                childParentDict[fval].add(val)
        self.childParentDict = childParentDict

    def buildHomologyMap(self, humanProteinList, mouseProteinList):

        # builds a map, between human/mouse data
        mapProteinSet = pd.merge(
            humanProteinList,
            mouseProteinList,
            on="homologene_group_id",
            suffixes=("_h", "_m"),
        )
        mapProteinSet = mapProteinSet.rename(
            columns={"protein_id_m": "protein_id"}
        )
        return mapProteinSet

    # the following function will be used to fetch the description of pathway
    def fetchPathwayIdDescription(self):
        idNameDict = {}
        reactome = selectAsDF(
            "select distinct reactome_id, name from reactome",
            ["reactome_id", "name"],
            self.db,
        )
        logging.debug(
            "(OlegDB.fetchPathwayIdDescription) Reactome IDs: {0}".format(
                reactome.shape[0]
            )
        )
        reactomeDict = reactome.set_index("reactome_id").T.to_dict("records")[
            0
        ]  # DataFrame to dictionary
        idNameDict.update(reactomeDict)

        kegg = selectAsDF(
            "select distinct kegg_pathway_id, kegg_pathway_name from kegg_pathway",
            ["kegg_pathway_id", "kegg_pathway_name"],
            self.db,
        )
        logging.debug(
            "(OlegDB.fetchPathwayIdDescription) KEGG pathway IDs: {0}".format(
                kegg.shape[0]
            )
        )
        keggDict = kegg.set_index("kegg_pathway_id").T.to_dict("records")[
            0
        ]  # DataFrame to dictionary
        idNameDict.update(keggDict)

        interpro = selectAsDF(
            "select distinct entry_ac,entry_name from interpro",
            ["entry_ac", "entry_name"],
            self.db,
        )
        logging.debug(
            "(OlegDB.fetchPathwayIdDescription) Interpro IDs: {0}".format(
                interpro.shape[0]
            )
        )
        interproDict = interpro.set_index("entry_ac").T.to_dict("records")[
            0
        ]  # DataFrame to dictionary
        idNameDict.update(interproDict)

        goa = selectAsDF(
            "select distinct go_id, name from go", ["go_id", "name"], self.db
        )
        logging.debug(
            "(OlegDB.fetchPathwayIdDescription) GO IDs: {0}".format(
                goa.shape[0]
            )
        )
        goaDict = goa.set_index("go_id").T.to_dict("records")[
            0
        ]  # DataFrame to dictionary
        idNameDict.update(goaDict)

        ppi = selectAsDF(
            "select distinct protein_id, name from protein",
            ["protein_id", "name"],
            self.db,
        )
        logging.debug(
            "(OlegDB.fetchPathwayIdDescription) Protein IDs: {0}".format(
                ppi.shape[0]
            )
        )
        ppiDict = ppi.set_index("protein_id").T.to_dict("records")[
            0
        ]  # DataFrame to dictionary
        idNameDict.update(ppiDict)

        mp = selectAsDF(
            "select distinct mp_term_id, name from mp_onto",
            ["mp_term_id", "name"],
            self.db,
        )
        logging.debug(
            "(OlegDB.fetchPathwayIdDescription) MP IDs: {0}".format(
                mp.shape[0]
            )
        )
        mpDict = mp.set_index("mp_term_id").T.to_dict("records")[
            0
        ]  # DataFrame to dictionary
        idNameDict.update(mpDict)
        return idNameDict

    # the following function will be used to assign the database source to each protein id
    def addDatabaseSourceToProteinId(self):
        idSourceDict = {}
        reactome = selectAsDF(
            "select distinct reactome_id, 'Reactome' as source from reactome",
            ["reactome_id", "source"],
            self.db,
        )
        logging.debug(
            "(OlegDB.addDatabaseSourceToProteinId) Reactome IDs: {0}".format(
                reactome.shape[0]
            )
        )
        reactomeDict = reactome.set_index("reactome_id").T.to_dict("records")[
            0
        ]  # DataFrame to dictionary
        idSourceDict.update(reactomeDict)

        kegg = selectAsDF(
            "select distinct kegg_pathway_id, 'Kegg' as source from kegg_pathway",
            ["kegg_pathway_id", "source"],
            self.db,
        )
        logging.debug(
            "(OlegDB.addDatabaseSourceToProteinId) KEGG pathway IDs: {0}".format(
                kegg.shape[0]
            )
        )
        keggDict = kegg.set_index("kegg_pathway_id").T.to_dict("records")[
            0
        ]  # DataFrame to dictionary
        idSourceDict.update(keggDict)

        interpro = selectAsDF(
            "select distinct entry_ac, 'Interpro' as source from interpro",
            ["entry_ac", "source"],
            self.db,
        )
        logging.debug(
            "(OlegDB.addDatabaseSourceToProteinId) Interpro IDs: {0}".format(
                interpro.shape[0]
            )
        )
        interproDict = interpro.set_index("entry_ac").T.to_dict("records")[
            0
        ]  # DataFrame to dictionary
        idSourceDict.update(interproDict)

        goa = selectAsDF(
            "select distinct go_id, 'GO' as source from go",
            ["go_id", "source"],
            self.db,
        )
        logging.debug(
            "(OlegDB.addDatabaseSourceToProteinId) GO IDs: {0}".format(
                goa.shape[0]
            )
        )
        goaDict = goa.set_index("go_id").T.to_dict("records")[
            0
        ]  # DataFrame to dictionary
        idSourceDict.update(goaDict)

        ppi = selectAsDF(
            "select distinct protein_id, 'PPI string' as source from protein",
            ["protein_id", "source"],
            self.db,
        )
        logging.debug(
            "(OlegDB.addDatabaseSourceToProteinId) Protein IDs: {0}".format(
                ppi.shape[0]
            )
        )
        ppiDict = ppi.set_index("protein_id").T.to_dict("records")[
            0
        ]  # DataFrame to dictionary
        idSourceDict.update(ppiDict)

        mp = selectAsDF(
            "select distinct mp_term_id, 'MP term' as source from mp_onto",
            ["mp_term_id", "source"],
            self.db,
        )
        logging.debug(
            "(OlegDB.addDatabaseSourceToProteinId) MP IDs: {0}".format(
                mp.shape[0]
            )
        )
        mpDict = mp.set_index("mp_term_id").T.to_dict("records")[
            0
        ]  # DataFrame to dictionary
        idSourceDict.update(mpDict)

        # the following function will fetch drug_id and drug_name from drug_name table.
        allDrugNames = selectAsDF(
            "select 'drug_'||drug_id AS col_id, drug_name from drug_name",
            ["col_id", "drug_name"],
            self.db,
        )
        logging.debug(
            "(OlegDB.addDatabaseSourceToProteinId) All Drug Ids: {0}".format(
                allDrugNames.shape[0]
            )
        )
        drugIdNameDict = allDrugNames.set_index("col_id").T.to_dict("records")[
            0
        ]  # DataFrame to dictionary
        idSourceDict.update(drugIdNameDict)
        return idSourceDict

    # the following function will be used to fetch the protein_id for the given symbols
    def fetchProteinIdForSymbol(self, symbolList):
        sql = (
            "SELECT distinct symbol, protein_id FROM protein WHERE symbol in ("
        )
        for symbol in symbolList[:-1]:
            sql = sql + "'" + symbol + "'" + ","
        sql = sql + "'" + symbolList[-1] + "')"
        # print (sql)
        symbolProtein = selectAsDF(sql, ["symbol", "protein_id"], self.db)
        logging.debug(
            "(OlegDB.fetchProteinIdForSymbol) Protein ID for Symbol: {0}".format(
                symbolProtein.shape[0]
            )
        )
        symbolProteinIdDict = symbolProtein.set_index("symbol").T.to_dict(
            "records"
        )[
            0
        ]  # DataFrame to dictionary
        return symbolProteinIdDict

    # the following function will fetch protein_ids for tax_id 9606.
    def fetchAllProteinIds(self):
        allProteinIds = selectAsDF(
            "select distinct protein_id from protein where tax_id=9606",
            ["protein_id"],
            self.db,
        )
        logging.debug(
            "(OlegDB.fetchAllProteinIds) All Protein Ids: {0}".format(
                allProteinIds.shape[0]
            )
        )
        return allProteinIds

    # the following function will be used to fetch symbol, name, and species for given protein_id
    def fetchSymbolForProteinId(self):
        proteinIdSymbol = selectAsDF(
            "select distinct protein_id, symbol from protein",
            ["protein_id", "symbol"],
            self.db,
        )
        logging.debug(
            "(OlegDB.fetchSymbolForProteinId) Protein Name for Id: {0}".format(
                proteinIdSymbol.shape[0]
            )
        )
        proteinIdSymbolDict = proteinIdSymbol.set_index(
            "protein_id"
        ).T.to_dict("records")[
            0
        ]  # DataFrame to dictionary
        return proteinIdSymbolDict

    # the following function with fetch all protein ids with negative class
    def fetchNegativeClassProteinIds(self):
        sql = """\
SELECT DISTINCT
    clinvar.protein_id
FROM
    clinvar
JOIN
    clinvar_phenotype ON clinvar.clinvar_phenotype_id = clinvar_phenotype.id
JOIN
    clinvar_phenotype_xref ON clinvar_phenotype.id = clinvar_phenotype_xref.clinvar_phenotype_id
WHERE 
    clinvar_phenotype_xref.source = 'OMIM'
    AND clinvar.clinical_significance IN (
'Pathogenic, Affects', 'Benign, protective, risk factor', 'Pathogenic/Likely pathogenic', 'Pathogenic/Likely pathogenic, other', 'Pathogenic, other',
'Affects', 'Pathogenic, other, protective', 'Conflicting interpretations of pathogenicity, Affects, association, other', 'Pathogenic/Likely pathogenic, drug response',
'Pathogenic, risk factor', 'risk factor', 'Pathogenic, association', 'Conflicting interpretations of pathogenicity, Affects, association, risk factor',
'Pathogenic/Likely pathogenic, risk factor', 'Affects, risk factor', 'Conflicting interpretations of pathogenicity, association, other, risk factor',
'Likely pathogenic, association', 'association, protective', 'Likely pathogenic, Affects', 'Pathogenic', 'Conflicting interpretations of pathogenicity, association',
'Pathogenic/Likely pathogenic, Affects, risk factor', 'Conflicting interpretations of pathogenicity, other, risk factor', 'association, risk factor',
'Benign, protective', 'Conflicting interpretations of pathogenicity, risk factor', 'Uncertain significance, protective', 'association', 'Uncertain significance, Affects',
'protective, risk factor', 'Pathogenic, association, protective', 'Pathogenic, protective', 'Likely pathogenic, other', 'Pathogenic, protective, risk factor',
'Benign, association, protective', 'Conflicting interpretations of pathogenicity, Affects', 'Benign/Likely benign, protective', 'protective')
"""
        negProteinIds = selectAsDF(sql, ["protein_id"], self.db)
        logging.debug(
            "(OlegDB.fetchNegativeClassProteinIds) Negative class Protein Ids: {0}".format(
                negProteinIds.shape[0]
            )
        )
        return negProteinIds


# Should have same methods as OlegDB.
class TCRD(Adapter):
    config_file = os.environ["HOME"] + "/.ProteinGraphML.yaml"
    # config_file = "/code/DBcreds.yaml"
    GTD = None
    mouseToHumanAssociation = None
    geneToDisease = None
    childParentDict = None

    db = None

    def __init__(self):
        self.load()

    def loadTotalProteinList(self):
        protein = selectAsDF(
            "SELECT DISTINCT id AS protein_id FROM protein",
            ["protein_id"],
            self.db,
        )
        logging.debug(
            "(TCRD.loadTotalProteinList) Human protein IDs returned: {0}".format(
                protein.shape[0]
            )
        )
        return protein

    def loadReactome(self, proteinFilter=None):
        reactome = selectAsDF(
            'SELECT protein_id, id_in_source AS reactome_id, name AS "evidence" FROM pathway WHERE pwtype = '
            "'Reactome'",
            ["protein_id", "reactome_id", "evidence"],
            self.db,
        )
        if proteinFilter is not None:
            reactome = reactome[reactome["protein_id"].isin(proteinFilter)]
        logging.debug(
            "(TCRD.loadReactome) Reactome rows returned: {0}".format(
                reactome.shape[0]
            )
        )
        return GraphEdge("protein_id", "reactome_id", data=reactome)

    def loadPPI(self, proteinFilter=None):
        stringDB = selectAsDF(
            "SELECT protein1_id, protein2_id, score AS combined_score FROM ppi WHERE ppitype = 'STRINGDB'",
            ["protein_id1", "protein_id2", "combined_score"],
            self.db,
        )
        if proteinFilter is not None:
            stringDB = stringDB[stringDB["protein_id1"].isin(proteinFilter)]
            stringDB = stringDB[stringDB["protein_id2"].isin(proteinFilter)]
        logging.debug(
            "(TCRD.loadPPI) STRING rows returned: {0}".format(
                stringDB.shape[0]
            )
        )
        return GraphEdge(
            "protein_id1", "protein_id2", "combined_score", stringDB
        )

    def loadKegg(self, proteinFilter=None):
        kegg = selectAsDF(
            "SELECT protein_id, SUBSTR(id_in_source, 6) AS kegg_pathway_id FROM pathway WHERE pwtype = 'KEGG'",
            ["protein_id", "kegg_pathway_id"],
            self.db,
        )
        if proteinFilter is not None:
            kegg = kegg[kegg["protein_id"].isin(proteinFilter)]
        logging.debug(
            "(TCRD.loadKegg) KEGG rows returned: {0}".format(kegg.shape[0])
        )
        return GraphEdge("protein_id", "kegg_pathway_id", data=kegg)

    def loadInterpro(self, proteinFilter=None):
        interpro = selectAsDF(
            "SELECT DISTINCT protein_id, value AS entry_ac FROM xref WHERE xtype = 'InterPro'",
            ["protein_id", "entry_ac"],
            self.db,
        )
        if proteinFilter is not None:
            interpro = interpro[interpro["protein_id"].isin(proteinFilter)]
        logging.debug(
            "(TCRD.loadInterpro) Interpro rows returned: {0}".format(
                interpro.shape[0]
            )
        )
        return GraphEdge("protein_id", "entry_ac", data=interpro)

    def loadPFAM(self, proteinFilter=None):
        pfam = selectAsDF(
            "SELECT DISTINCT protein_id, value AS entry_ac FROM xref WHERE xtype = 'Pfam'",
            ["protein_id", "entry_ac"],
            self.db,
        )
        if proteinFilter is not None:
            pfam = pfam[pfam["protein_id"].isin(proteinFilter)]
        logging.debug(
            "(TCRD.loadPFAM) Pfam rows returned: {0}".format(pfam.shape[0])
        )
        return GraphEdge("protein_id", "entry_ac", data=pfam)

    def loadPROSITE(self, proteinFilter=None):
        prosite = selectAsDF(
            "SELECT DISTINCT protein_id, value AS entry_ac FROM xref WHERE xtype = 'PROSITE'",
            ["protein_id", "entry_ac"],
            self.db,
        )
        if proteinFilter is not None:
            prosite = prosite[prosite["protein_id"].isin(proteinFilter)]
        logging.debug(
            "(TCRD.loadPROSITE) Prosite rows returned: {0}".format(
                prosite.shape[0]
            )
        )
        return GraphEdge("protein_id", "entry_ac", data=prosite)

    def loadGo(self, proteinFilter=None):
        goa = selectAsDF(
            "SELECT protein_id, go_id FROM goa",
            ["protein_id", "go_id"],
            self.db,
        )
        if proteinFilter is not None:
            goa = goa[goa["protein_id"].isin(proteinFilter)]
        logging.debug(
            "(TCRD.loadGo) GO rows returned: {0}".format(goa.shape[0])
        )
        return GraphEdge("protein_id", "go_id", data=goa)

    def loadOMIM(self, proteinFilter=None):
        sql = """\
SELECT DISTINCT 
    clinvar.protein_id,
    protein.sym,
    clinvar_phenotype.id AS clinvar_phenotype_id,
    clinvar.clinical_significance,
    clinvar_phenotype.name,
    omim.mim AS mim_id,
    omim.title AS mim_title
FROM 
    clinvar, clinvar_phenotype, clinvar_phenotype_xref, protein, omim
WHERE
    clinvar.clinvar_phenotype_id = clinvar_phenotype.id
    AND clinvar_phenotype.id = clinvar_phenotype_xref.clinvar_phenotype_id  
    AND clinvar_phenotype_xref.source = 'OMIM'
    AND clinvar_phenotype_xref.value = omim.mim
    AND clinvar.protein_id = protein.id
    AND clinvar.clinical_significance NOT IN (
        'Uncertain significance',
        'Uncertain significance, association',
        'Uncertain significance, other',
        'other, risk factor',
        'other')
"""
        omim = selectAsDF(
            sql,
            [
                "protein_id",
                "symbol",
                "phenotype_id",
                "clinical_significance",
                "phenotype.name",
                "mim_id",
                "mim_title",
            ],
            self.db,
        )

        if proteinFilter is not None:
            omim = omim[omim["protein_id"].isin(proteinFilter)]
        logging.debug(
            "(TCRD.loadOMIM) OMIM rows returned: {0}".format(omim.shape[0])
        )
        return GraphEdge("protein_id", "mim_id", data=omim)

    # static features
    def loadGTEX(self):
        # Average = (F+M)/2, where F and M are medians.
        gtex = selectAsDF(
            "SELECT protein_id, CAST(AVG(tpm) AS DECIMAL(5,3)) AS median_tpm, tissue AS tissue_type_detail FROM gtex "
            "GROUP BY protein_id, tissue",
            ["protein_id", "median_tpm", "tissue_type_detail"],
            self.db,
        )
        logging.debug(
            "({0}.loadGTEX) gtex rows: {1}".format(
                type(self).__name__, gtex.shape[0]
            )
        )
        gtex.median_tpm = gtex.median_tpm.astype(float)  # Why necessary?
        gtex.info()
        return gtex

    def loadCCLE(self):
        ccle = selectAsDF(
            "SELECT protein_id, cell_id, tissue, number_value AS expression FROM expression WHERE etype = 'ccle'",
            ["protein_id", "cell_id", "tissue", "expression"],
            self.db,
        )
        logging.debug(
            "({0}.loadCCLE) ccle rows: {1}".format(
                type(self).__name__, ccle.shape[0]
            )
        )
        ccle.expression = ccle.expression.astype(float)  # Why necessary?
        return ccle

    def loadLINCS(self):
        lincs = selectAsDF(
            "SELECT protein_id, CONCAT(pert_dcid, ':', cellid) AS col_id, zscore FROM lincs",
            ["protein_id", "col_id", "zscore"],
            self.db,
        )
        logging.debug(
            "({0}.loadLINCS) lincs rows: {1}".format(
                type(self).__name__, lincs.shape[0]
            )
        )
        lincs.zscore = lincs.zscore.astype(float)  # Why necessary?
        lincs.info()
        return lincs

    def loadHPA(self):
        # ## IN TCRD, cell_id all NULL? hpa = selectAsDF("SELECT protein_id, CONCAT(tissue, '.', cell_id) AS col_id,
        # qual_value FROM expression WHERE etype = 'HPA' AND evidence IN ('Approved', 'Supported')", ["protein_id",
        # "col_id", "level"], self.db)
        hpa = selectAsDF(
            "SELECT DISTINCT protein_id, tissue AS col_id, qual_value AS level FROM expression WHERE etype = 'HPA' "
            "AND evidence IN ('Approved', 'Supported')",
            ["protein_id", "col_id", "level"],
            self.db,
        )
        logging.debug(
            "({0}.loadHPA) hpa rows: {0}".format(
                type(self).__name__, hpa.shape[0]
            )
        )
        hpa.info()
        return hpa

    #
    def load(self):
        with open(self.config_file, "r") as stream:
            try:
                credentials = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                logging.error(
                    "DB credentials not found in {0}: {1}".format(
                        self.config_file, str(exc)
                    )
                )

        self.db = Database()
        self.db.bind(
            provider="mysql",
            user=credentials["tcrd_user"],
            password=credentials["tcrd_password"],
            host=credentials["tcrd_host"],
            database=credentials["tcrd_database"],
        )
        logging.debug(
            "(TCRD.load) Connected to db (%s): %s:%s:%s"
            % (
                self.db.provider_name,
                credentials["tcrd_host"],
                credentials["tcrd_database"],
                credentials["tcrd_user"],
            )
        )
        self.db.generate_mapping(create_tables=False)

        # hack ... saving the (DB) like this
        db = self.db
        #
        humanProteinList = selectAsDF(
            "SELECT id AS hid, groupid AS homologene_group_id, taxid AS tax_id, protein_id  FROM homologene WHERE "
            "taxid = 9606",
            ["hid", "homologene_group_id", "tax_id", "protein_id"],
            db,
        )
        logging.debug(
            "(TCRD.load) humanProteinList rows: %d"
            % (humanProteinList.shape[0])
        )
        logging.debug(
            "(TCRD.load) humanProteinList.protein_id.nunique(): %d"
            % (humanProteinList.protein_id.nunique())
        )
        logging.debug(
            "(TCRD.load) humanProteinList.homologene_group_id.nunique(): %d"
            % (humanProteinList.homologene_group_id.nunique())
        )

        mouseProteinList = selectAsDF(
            "SELECT id AS hid, groupid AS homologene_group_id, taxid AS tax_id, nhprotein_id AS protein_id_m FROM "
            "homologene WHERE taxid = 10090",
            ["hid", "homologene_group_id", "tax_id", "protein_id_m"],
            db,
        )
        logging.debug(
            "(TCRD.load) mouseProteinList rows: %d"
            % (mouseProteinList.shape[0])
        )
        logging.debug(
            "(TCRD.load) mouseProteinList.protein_id_m.nunique(): %d"
            % (mouseProteinList.protein_id_m.nunique())
        )
        logging.debug(
            "(TCRD.load) mouseProteinList.homologene_group_id.nunique(): %d"
            % (mouseProteinList.homologene_group_id.nunique())
        )

        mousePhenotype = selectAsDF(
            "SELECT DISTINCT nhprotein_id AS protein_id_m, term_id AS mp_term_id, p_value, effect_size, "
            "procedure_name, parameter_name, gp_assoc AS association FROM phenotype WHERE ptype = 'IMPC'",
            [
                "protein_id_m",
                "mp_term_id",
                "p_value",
                "effect_size",
                "procedure_name",
                "parameter_name",
                "association",
            ],
            db,
        )
        # mousePhenotype.info() #DEBUG
        logging.debug(
            "(TCRD.load) mousePhenotype rows: %d" % (mousePhenotype.shape[0])
        )
        logging.debug(
            "(TCRD.load) mousePhenotype.protein_id_m.nunique(): %d"
            % (mousePhenotype.protein_id_m.nunique())
        )

        mpOnto = selectAsDF(
            "SELECT mpid AS mp_term_id, parent_id, name FROM mpo",
            ["mp_term_id", "parent_id", "name"],
            db,
        )
        logging.debug("(TCRD.load) mpOnto rows: %d" % (mpOnto.shape[0]))
        self.saveNameMap("MP_ontology", "mp_term_id", "name", mpOnto)

        mouseToHumanMap = pd.merge(
            humanProteinList,
            mouseProteinList,
            on="homologene_group_id",
            suffixes=("_h", "_m"),
        )

        combinedSet = pd.merge(
            mouseToHumanMap, mousePhenotype, on="protein_id_m", copy=False
        )

        mouseToHumanAssociation = combinedSet[
            ["protein_id", "mp_term_id", "association"]
        ].drop_duplicates()
        logging.debug(
            "(TCRD.load) mouseToHumanAssociation rows: %d"
            % (mouseToHumanAssociation.shape[0])
        )

        def getVal(row):
            return depthMap[row["mp_term_id"]]

        depthMap = generateDepthMap(mpOnto)
        mpOnto["level"] = mpOnto.apply(getVal, axis=1)
        mpOnto = mpOnto[mpOnto["level"] > 1]  # remove the single level stuff
        geneToDisease = pd.merge(
            mouseToHumanAssociation, mpOnto, on="mp_term_id", copy=False
        )
        logging.debug(
            "(TCRD.load) geneToDisease rows: %d" % (geneToDisease.shape[0])
        )
        # logging.debug("(TCRD.load) geneToDisease.info():"); geneToDisease.info()
        logging.debug(
            "(TCRD.load) geneToDisease.protein_id.nunique(): %d"
            % (geneToDisease.protein_id.nunique())
        )

        # we could extract this piece layer
        self.geneToDisease = GraphEdge(
            "mp_term_id",
            "protein_id",
            edge="association",
            data=mouseToHumanAssociation,
        )

        parentHierarchy = GraphEdge(
            "mp_term_id",
            "parent_id",
            edge=None,
            data=geneToDisease[["mp_term_id", "parent_id"]],
        )
        parentHierarchy.setDirected()
        self.phenotypeHierarchy = parentHierarchy

        # this child dict saves parents in reverse order, so that you can look them up directly
        childParentDict = {}
        for fval, val in zip(
            self.phenotypeHierarchy.data["mp_term_id"],
            self.phenotypeHierarchy.data["parent_id"],
        ):
            if fval not in childParentDict.keys():
                childParentDict[fval] = set([val])
            else:
                childParentDict[fval].add(val)
        self.childParentDict = childParentDict

    def buildHomologyMap(self, humanProteinList, mouseProteinList):

        # builds a map, between human/mouse data
        mapProteinSet = pd.merge(
            humanProteinList,
            mouseProteinList,
            on="homologene_group_id",
            suffixes=("_h", "_m"),
        )
        mapProteinSet = mapProteinSet.rename(
            columns={"protein_id_m": "protein_id"}
        )
        return mapProteinSet

    # the following function will be used to fetch the description of pathway
    def fetchPathwayIdDescription(self):
        idNameDict = {}
        reactome = selectAsDF(
            "SELECT DISTINCT id_in_source AS reactome_id, name FROM pathway WHERE pwtype = 'Reactome'",
            ["reactome_id", "name"],
            self.db,
        )
        logging.debug(
            "(TCRD.fetchPathwayIdDescription) reactome rows: {0}".format(
                reactome.shape[0]
            )
        )
        reactomeDict = reactome.set_index("reactome_id").T.to_dict("records")[
            0
        ]  # DataFrame to dictionary
        idNameDict.update(reactomeDict)

        kegg = selectAsDF(
            "SELECT DISTINCT SUBSTR(id_in_source, 6) AS kegg_pathway_id, name AS kegg_pathway_name FROM pathway WHERE "
            "pwtype = 'KEGG'",
            ["kegg_pathway_id", "kegg_pathway_name"],
            self.db,
        )
        logging.debug(
            "(TCRD.fetchPathwayIdDescription) kegg rows: {0}".format(
                kegg.shape[0]
            )
        )
        keggDict = kegg.set_index("kegg_pathway_id").T.to_dict("records")[
            0
        ]  # DataFrame to dictionary
        idNameDict.update(keggDict)

        interpro = selectAsDF(
            "SELECT DISTINCT value AS entry_ac, xtra AS entry_name FROM xref WHERE xtype = 'InterPro'",
            ["entry_ac", "entry_name"],
            self.db,
        )
        logging.debug(
            "(TCRD.fetchPathwayIdDescription) interpro rows: {0}".format(
                interpro.shape[0]
            )
        )
        interproDict = interpro.set_index("entry_ac").T.to_dict("records")[
            0
        ]  # DataFrame to dictionary
        idNameDict.update(interproDict)

        goa = selectAsDF(
            "SELECT DISTINCT go_id, go_term AS name FROM goa",
            ["go_id", "name"],
            self.db,
        )
        logging.debug(
            "(TCRD.fetchPathwayIdDescription) goa rows: {0}".format(
                goa.shape[0]
            )
        )
        goaDict = goa.set_index("go_id").T.to_dict("records")[
            0
        ]  # DataFrame to dictionary
        idNameDict.update(goaDict)

        ppi = selectAsDF(
            "SELECT DISTINCT id AS protein_id, description AS name FROM protein",
            ["protein_id", "name"],
            self.db,
        )
        logging.debug(
            "(TCRD.fetchPathwayIdDescription) ppi rows: {0}".format(
                ppi.shape[0]
            )
        )
        ppiDict = ppi.set_index("protein_id").T.to_dict("records")[
            0
        ]  # DataFrame to dictionary
        idNameDict.update(ppiDict)

        mp = selectAsDF(
            "SELECT DISTINCT mpid AS mp_term_id, name FROM mpo",
            ["mp_term_id", "name"],
            self.db,
        )
        logging.debug(
            "(TCRD.fetchPathwayIdDescription) mp rows: {0}".format(mp.shape[0])
        )
        mpDict = mp.set_index("mp_term_id").T.to_dict("records")[
            0
        ]  # DataFrame to dictionary
        idNameDict.update(mpDict)
        return idNameDict

    # the following function will be used to assign the database source to each protein id
    def addDatabaseSourceToProteinId(self):
        idSourceDict = {}
        reactome = selectAsDF(
            "SELECT DISTINCT id_in_source AS reactome_id, 'Reactome' as source FROM pathway WHERE pwtype = 'Reactome'",
            ["reactome_id", "source"],
            self.db,
        )
        logging.debug(
            "(TCRD.addDatabaseSourceToProteinId) Reactome IDs: {0}".format(
                reactome.shape[0]
            )
        )
        reactomeDict = reactome.set_index("reactome_id").T.to_dict("records")[
            0
        ]  # DataFrame to dictionary
        idSourceDict.update(reactomeDict)

        kegg = selectAsDF(
            "SELECT DISTINCT SUBSTR(id_in_source, 6) AS kegg_pathway_id, 'Kegg' as source FROM pathway WHERE pwtype = "
            "'KEGG'",
            ["kegg_pathway_id", "source"],
            self.db,
        )
        logging.debug(
            "(TCRD.addDatabaseSourceToProteinId) KEGG pathway IDs: {0}".format(
                kegg.shape[0]
            )
        )
        keggDict = kegg.set_index("kegg_pathway_id").T.to_dict("records")[
            0
        ]  # DataFrame to dictionary
        idSourceDict.update(keggDict)

        interpro = selectAsDF(
            "SELECT DISTINCT value AS entry_ac, 'Interpro' as source FROM xref WHERE xtype = 'InterPro'",
            ["entry_ac", "source"],
            self.db,
        )
        logging.debug(
            "(TCRD.addDatabaseSourceToProteinId) Interpro IDs: {0}".format(
                interpro.shape[0]
            )
        )
        interproDict = interpro.set_index("entry_ac").T.to_dict("records")[
            0
        ]  # DataFrame to dictionary
        idSourceDict.update(interproDict)

        goa = selectAsDF(
            "SELECT DISTINCT go_id, 'GO' as source FROM goa",
            ["go_id", "source"],
            self.db,
        )
        logging.debug(
            "(TCRD.addDatabaseSourceToProteinId) GO IDs: {0}".format(
                goa.shape[0]
            )
        )
        goaDict = goa.set_index("go_id").T.to_dict("records")[
            0
        ]  # DataFrame to dictionary
        idSourceDict.update(goaDict)

        ppi = selectAsDF(
            "SELECT DISTINCT id AS protein_id, 'PPI string' as source FROM protein",
            ["protein_id", "source"],
            self.db,
        )
        logging.debug(
            "(TCRD.addDatabaseSourceToProteinId) Protein IDs: {0}".format(
                ppi.shape[0]
            )
        )
        ppiDict = ppi.set_index("protein_id").T.to_dict("records")[
            0
        ]  # DataFrame to dictionary
        idSourceDict.update(ppiDict)

        mp = selectAsDF(
            "SELECT DISTINCT mpid AS mp_term_id, 'MP term' as source FROM mpo",
            ["mp_term_id", "source"],
            self.db,
        )
        logging.debug(
            "(TCRD.addDatabaseSourceToProteinId) MP IDs: {0}".format(
                mp.shape[0]
            )
        )
        mpDict = mp.set_index("mp_term_id").T.to_dict("records")[
            0
        ]  # DataFrame to dictionary
        idSourceDict.update(mpDict)

        # the following function will fetch drug_id and drug_name from drug_name table.
        allDrugNames = selectAsDF(
            "select 'drug_'||dcid AS col_id, drug from drug_activity",
            ["col_id", "drug_name"],
            self.db,
        )
        logging.debug(
            "(TCRD.addDatabaseSourceToProteinId) All Drug Ids: {0}".format(
                allDrugNames.shape[0]
            )
        )
        drugIdNameDict = allDrugNames.set_index("col_id").T.to_dict("records")[
            0
        ]  # DataFrame to dictionary
        idSourceDict.update(drugIdNameDict)
        return idSourceDict

    def fetchProteinIdForSymbol(self, symbolList):
        sql = "SELECT DISTINCT sym AS symbol, id AS protein_id FROM protein WHERE sym in ("
        for symbol in symbolList[:-1]:
            sql = sql + "'" + symbol + "'" + ","
        sql = sql + "'" + symbolList[-1] + "')"
        symbolProtein = selectAsDF(sql, ["symbol", "protein_id"], self.db)
        logging.debug(
            "(TCRD.fetchProteinIdForSymbol) symbolProtein rows: {0}".format(
                symbolProtein.shape[0]
            )
        )
        symbolProteinIdDict = symbolProtein.set_index("symbol").T.to_dict(
            "records"
        )[
            0
        ]  # DataFrame to dictionary
        return symbolProteinIdDict

    def fetchAllProteinIds(self):
        allProteinIds = selectAsDF(
            "SELECT DISTINCT id AS protein_id FROM protein",
            ["protein_id"],
            self.db,
        )
        logging.debug(
            "(TCRD.fetchProteinIdForSymbol) allProteinIds rows: {0}".format(
                allProteinIds.shape[0]
            )
        )
        return allProteinIds

    def fetchSymbolForProteinId(self):
        proteinIdSymbol = selectAsDF(
            "SELECT DISTINCT id AS protein_id, sym AS symbol FROM protein",
            ["protein_id", "symbol"],
            self.db,
        )
        logging.debug(
            "(TCRD.fetchProteinIdForSymbol) proteinIdSymbol rows: {0}".format(
                proteinIdSymbol.shape[0]
            )
        )
        proteinIdSymbolDict = proteinIdSymbol.set_index(
            "protein_id"
        ).T.to_dict("records")[
            0
        ]  # DataFrame to dictionary
        return proteinIdSymbolDict

    def fetchUniprotForProteinId(self):
        proteinIdUniprot = selectAsDF(
            "SELECT DISTINCT id AS protein_id, uniprot AS uniprot_id FROM protein",
            ["protein_id", "uniprot_id"],
            self.db,
        )
        logging.debug(
            "(TCRD.fetchUniprotForProteinId) proteinIdUniprot rows: {0}".format(
                proteinIdUniprot.shape[0]
            )
        )
        proteinIdUniprotDict = proteinIdUniprot.set_index(
            "protein_id"
        ).T.to_dict("records")[
            0
        ]  # DataFrame to dictionary
        return proteinIdUniprotDict

    # the following function with fetch all protein ids with negative class
    def fetchNegativeClassProteinIds(self):
        sql = """\
SELECT DISTINCT 
    clinvar.protein_id
FROM
    clinvar
JOIN
    clinvar_phenotype ON clinvar.clinvar_phenotype_id = clinvar_phenotype.id
JOIN
    clinvar_phenotype_xref ON clinvar_phenotype.id = clinvar_phenotype_xref.clinvar_phenotype_id
WHERE 
    clinvar_phenotype_xref.source = 'OMIM'
    AND clinvar.clinical_significance != 'Uncertain significance'
"""
        negProteinIds = selectAsDF(sql, ["protein_id"], self.db)
        logging.debug(
            "(TCRD.fetchNegativeClassProteinIds) Negative class Protein Ids: {0}".format(
                negProteinIds.shape[0]
            )
        )
        return negProteinIds
