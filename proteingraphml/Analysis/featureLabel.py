from .MLTools.MetapathFeatures import (
    ProteinInteractionNode,
    KeggNode,
    ReactomeNode,
    GoNode,
    InterproNode,
)


def getValueForId(label, inputValue, extractKey, DB):
    results = DB[DB[label] == inputValue]
    if len(results) == 0:
        return None
    else:
        return results.iloc[0][extractKey]


def convertLabels(featureLabels, adapter, selectAsDF, type="graph"):

    # given a list, map them to real things, also w/ a DB adapter
    # print('beans')

    protein = selectAsDF(
        "select name,symbol,protein_id from protein",
        ["name", "symbol", "protein_id"],
        adapter.db,
    )
    # MPONTO = selectAsDF("select * ")
    kegg = selectAsDF(
        "select kegg_pathway_id,kegg_pathway_name from kegg_pathway",
        ["kegg_pathway_id", "kegg_pathway_name"],
        adapter.db,
    )
    mpOnto = selectAsDF(
        "select mp_term_id,name from mp_onto",
        ["mp_term_id", "name"],
        adapter.db,
    )

    newMap = {}
    for item in featureLabels:
        newValue = item
        prefix = ""
        if ProteinInteractionNode.isThisNode(item):  # is a protein node:...

            if type != "graph":
                prefix = "PPI:"

            print(protein)
            newValue = "{0}{1}".format(
                prefix, getValueForId("protein_id", item, "symbol", protein)
            )

        # name = getValueForId("kegg_pathway_id",value,"kegg_pathway_name",kegg)
        if KeggNode.isThisNode(item):
            newValue = getValueForId(
                "kegg_pathway_id", item, "kegg_pathway_name", kegg
            )

        if isinstance(item, str) and item[:3] == "MP_":
            newValue = getValueForId("mp_term_id", item, "name", mpOnto)

        newMap[item] = newValue
        # print(newValue)

    return newMap


##OLD visualize CODE! MAKES BAR CHART OF FEATURES

"""
df = pd.DataFrame(data.most_common(), columns=['feature', 'gain'])
plt.figure()
df['gain'] = (df['gain']/sum(df['gain'][:20]))
df['feature'] = df['feature'].map(processFeature)
r = df.head(20).plot( kind='barh',title=TITLE, x='feature', y='gain',color='tomato', legend=False, figsize=(10, 12))
r.set_xlabel('Importance')
r.set_ylabel('Features')
r.invert_yaxis()

r.figure.savefig(FILETITLE+'.png',bbox_inches='tight')

#fig.suptitle('bold figure suptitle', fontsize=14, fontweight='bold')

#r.suptitle(fontweight='bold')
"""


# OLD PROTEIN CODE... FOR CREATING NICE HUMAN READABLE LABELS


# def getValueForId(label,inputValue,extractKey,DB):
#     results = DB[DB[label] == inputValue]
#     if len(results) == 0:
#         return None
#     else:
#         return results.iloc[0][extractKey]

# def processFeature(value):

#     replace = value
#     if value[:3] == "pp.":
#         replace = "PPI:"+protein[protein.protein_id == int(value[3:])].iloc[0]['symbol']

#     isDrug = re.compile("\\d+:[A-Z]")

#     if isDrug.match(value):

#         ID = value[:value.find(':')]
#         CELL_ID = value[value.find(':'):]
#         name = getValueForId("drug_id",ID,"drug_name",drugname)

#         if name is None:
#             name = ID

#         replace = name + CELL_ID + " signature"

#     if value.find('_') > 0:
#         #query to make sure the tissue exists? IDK
#         replace = value[value.find('_')+1:] + " " + "("+value[:value.find('_')]+")"
#     elif len(ccle[ccle.cell_id == value]) > 0:
#         replace = "expression in "+value

#     if value[:3] == "hsa":
#         name = getValueForId("kegg_pathway_id",value,"kegg_pathway_name",kegg)
#         if name is None:
#             replace = value
#         else:
#             replace = name

#     return replace
