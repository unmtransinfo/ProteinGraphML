from pony.orm import *
import pandas as pd

# included these here for now
@db_session
def fetch(sql, db):
    return db.select(sql)


@db_session
def runDB(db):
    data = db.select("select * from mp_onto")
    return len(data)


@db_session
def selectAsDF(sql, columns, db):
    return pd.DataFrame(fetch(sql, db), columns=columns)


def generateDepthMap(mpOnto):
    treeMap = {}
    depthMap = {}
    for index, row in mpOnto.iterrows():
        currentId = row["mp_term_id"]
        parentId = row["parent_id"]
        if parentId is None:
            treeMap[currentId] = None
        else:
            treeMap[currentId] = parentId

    for index, row in mpOnto.iterrows():
        currentId = row["mp_term_id"]
        traverser = currentId
        depth = -1
        while traverser != None:
            traverser = treeMap[traverser]
            depth += 1
        depthMap[currentId] = depth
    return depthMap


def attachColumn(originDF, newDF, attachment):
    return pd.merge(originDF, newDF, on=attachment, copy=False)
