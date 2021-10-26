import itertools
import logging
import pandas as pd

from .nodes import ProteinInteractionNode


def getMetapaths(proteinGraph, start):

    children = getChildren(proteinGraph.graph, start)

    if (
        start in proteinGraph.childParentDict.keys()
    ):  # if we've got parents, lets remove them from this search
        children = list(
            set(children) - set(proteinGraph.childParentDict[start])
        )

    proteinMap = {True: set(), False: set()}
    for c in children:
        p = filterNeighbors(proteinGraph.graph, c, True)
        n = filterNeighbors(proteinGraph.graph, c, False)
        posPaths = len(p)
        negPaths = len(n)

        for pid in p:
            proteinMap[True].add(pid)

        for pid in n:
            proteinMap[False].add(pid)

    return proteinMap


# new graph stuff, things below have been removed
def filterNeighbors(graph, start, association):  # hard coded ... "association"
    return [
        a
        for a in graph.adj[start]
        if "association" in graph.edges[(start, a)].keys()
        and graph.edges[(start, a)]["association"] == association
    ]


def getChildren(graph, start):  # hard coded ... "association"
    return [
        a
        for a in graph.adj[start]
        if "association" not in graph.edges[(start, a)].keys()
    ]


def getTrainingProteinIds(disease, proteinGraph):
    """
    This function returns the protein ids for True and False labels.
    """
    paths = getMetapaths(
        proteinGraph, disease
    )  # a dictionary with 'True' and 'False' as keys and protein_id as values
    return paths[True], paths[False]


def metapathFeatures(
    disease,
    proteinGraph,
    featureList,
    idDescription,
    staticFeatures=None,
    staticDir=None,
    test=False,
    loadedLists=None,
):
    # we compute a genelist....
    # get the proteins
    # for each of the features, compute their metapaths, given an object, and graph+list... then they get joined
    # print(len(proteinGraph.graph.nodes))

    G = proteinGraph.graph  # this is our networkx api

    if loadedLists is not None:
        trueP = loadedLists[True]
        falseP = loadedLists[False]
        try:
            unknownP = loadedLists["unknown"]
        except:
            unknownP = []
    else:
        paths = getMetapaths(
            proteinGraph, disease
        )  # a dictionary with 'True' and 'False' as keys and protein_id as values
        trueP = paths[True]
        falseP = paths[False]
        unknownP = []

    logging.info(
        "(metapathFeatures) PREPARING TRUE ASSOCIATIONS: {0}".format(
            len(trueP)
        )
    )
    logging.info(
        "(metapathFeatures) PREPARING FALSE ASSOCIATIONS: {0}".format(
            len(falseP)
        )
    )
    logging.info(
        "(metapathFeatures) PREPARING UNKNOWN ASSOCIATIONS: {0}".format(
            len(unknownP)
        )
    )
    logging.info("(metapathFeatures) NODES IN GRAPH: {0}".format(len(G.nodes)))
    logging.info("(metapathFeatures) EDGES IN GRAPH: {0}".format(len(G.edges)))

    proteinNodes = [
        pro for pro in list(G.nodes) if ProteinInteractionNode.isThisNode(pro)
    ]  # if isinstance(pro,int)] # or isinstance(pro,np.integer)]

    if len(proteinNodes) == 0:
        raise Exception("No protein nodes detected in graph")

    logging.info(
        "(metapathFeatures) DETECTED PROTEINS: {0}".format(len(proteinNodes))
    )

    nodeListPairs = []
    for n in featureList:
        nodeListPairs.append(
            (n, [nval for nval in list(G.nodes) if n.isThisNode(nval)])
        )

    metapaths = []
    flog = "metapath_features.log"
    logging.info(
        "(metapathFeatures) Metapath features logfile: {0}".format(flog)
    )
    fh = open(flog, "w")  # file to save nodes used for metapaths
    for pair in nodeListPairs:
        nodes = pair[1]
        nonTrueAssociations = set(proteinNodes) - trueP
        # print(len(G.nodes), len(nodes), len(trueP), len(nonTrueAssociations))
        METAPATH = pair[0].computeMetapaths(
            G, nodes, trueP, nonTrueAssociations, idDescription, fh
        )
        METAPATH = (METAPATH - METAPATH.mean()) / METAPATH.std()
        logging.info(
            "(metapathFeatures) METAPATH FRAME {0}x{1} for {2}".format(
                METAPATH.shape[0], METAPATH.shape[1], pair[0]
            )
        )
        metapaths.append(METAPATH)
    fh.close()

    if test:
        fullList = list(proteinNodes)
        df = pd.DataFrame(fullList, columns=["protein_id"])
        df = df.set_index("protein_id")
    else:
        if len(unknownP) == 0:
            fullList = list(itertools.product(trueP, [1])) + list(
                itertools.product(falseP, [0])
            )
        else:
            fullList = (
                list(itertools.product(trueP, [1]))
                + list(itertools.product(falseP, [0]))
                + list(itertools.product(unknownP, [-1]))
            )
        df = pd.DataFrame(fullList, columns=["protein_id", "Y"])
        df = df.set_index("protein_id")

    for metapathframe in metapaths:
        # YOU CAN USE THESE TO GET A SUM IF NEED BE
        # print(metapathframe.shape)
        # print(sum(metapathframe.sum(axis=1)))

        df = df.join(metapathframe, on="protein_id")

    if staticFeatures is not None:
        df = joinStaticFeatures(df, staticFeatures, staticDir)

    return df


def joinStaticFeatures(df, features, datadir):
    for feature in features:
        try:  # newer, TSVs
            df_this = pd.read_csv(datadir + "/" + feature + ".tsv", "\t")
        except:  # older, CSVs
            df_this = pd.read_csv(datadir + "/" + feature + ".csv")
        #
        df_this = df_this.set_index("protein_id")
        df_this = df_this.drop(df_this.columns[0], axis=1)
        #
        if (
            feature == "gtex" or feature == "ccle"
        ):  # Kludge: all normed but hpa.
            df_this = (df_this - df_this.mean()) / df_this.std()
        df = df.join(df_this, on="protein_id")
    return df
