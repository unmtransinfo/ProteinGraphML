from threading import Thread
from tqdm import tqdm
import pandas as pd
import json

def createProteinMatrix(graph, indexColumn, columnIds,outputdir="./cql/config/dataset",datasetName="Protein"):
    results = []
    columns = columnIds
    for i in tqdm(indexColumn,desc=datasetName):
        q = "MATCH (p1:Protein{protein_id:%d})-[:STRING]->(p2:Protein) RETURN p2.protein_id as id" % (i)
        data = graph.run(q).data()
        matches = dict.fromkeys([v["id"] for v in data], 1)

        record = {"protein_id": i}
        for e in columns:
            if e in matches.keys():
                record[e] = 1
            else:
                record[e] = 0
        results.append(record)
    df = pd.DataFrame(results)
    df.to_csv(f"{outputdir}/{datasetName}.csv", index=False, encoding="utf-8")


def getInterPro(graph,index,ipr,outputdir="./cql/config/dataset",datasetName="INTERPRO"):
    entryAcs = ipr
    fullSet = []
    for i in tqdm(index,desc=datasetName):
        record = {}
        q = "MATCH (p:Protein{protein_id: %d})-->(k:INTERPRO) RETURN  k.entry_ac as entry_ac" % (i)
        data = graph.run(q).data()
        record = dict.fromkeys(entryAcs,0)
        record.update({"protein_id":int(i)})
        if data:
            [record.update({e["entry_ac"]:int(1)}) for e in data if e["entry_ac"] in entryAcs]
            fullSet.append(record)
        else:
            fullSet.append(record)
    df = pd.DataFrame(fullSet)
    df.to_csv(f"{outputdir}/{datasetName}.csv", index=False, encoding="utf-8")


def getKeggPathWays(graph,index,hsa,outputdir="./cql/config/dataset",datasetName="KEGG"):
    pathwayIds = hsa
    fullSet = []
    for i in tqdm(index,desc=datasetName):
        record = {}
        q = "MATCH (p:Protein{protein_id: %d})-->(k:KEGG) RETURN  k.kegg_pathway_id as hsa" % (i)
        data = graph.run(q).data()
        record = dict.fromkeys(pathwayIds,0)
        record.update({"protein_id":int(i)})
        if data:
            [record.update({e["hsa"]:int(1)}) for e in data if e["hsa"] in pathwayIds]
            fullSet.append(record)
        else:
            fullSet.append(record)
    df = pd.DataFrame(fullSet)
    df.to_csv(f"{outputdir}/{datasetName}.csv", index=False, encoding="utf-8")


def getReactome(graph,index,rHSA,outputdir="./cql/config/dataset",datasetName="KEGG"):
    fullSet = []
    reactomeIds = rHSA
    for i in tqdm(index,desc=datasetName):
        q = "MATCH (p:Protein{protein_id: %d})-->(k:REACTOME) RETURN  k.reactome_id as reactome_id" % (i)
        data = graph.run(q).data()
        record = dict.fromkeys(reactomeIds,0)
        record.update({"protein_id":int(i)})
        if data:
            [record.update({e["reactome_id"]:1}) for e in data if e["reactome_id"] in reactomeIds]
            fullSet.append(record)
        else:
            fullSet.append(record)
    df = pd.DataFrame(fullSet)
    df.to_csv(f"{outputdir}/{datasetName}.csv", index=False, encoding="utf-8")


def getGO(graph,index,go,outputdir="./cql/config/dataset",datasetName="GO"):
    fullSet = []
    goIds = go
    for i in tqdm(index,desc=datasetName):
        q = "MATCH (p:Protein{protein_id: %d})-->(k:GO) RETURN  k.go_id as go_id" % (i)
        data = graph.run(q).data()
        record = dict.fromkeys(goIds,0)
        record.update({"protein_id":int(i)})
        if data:
            [record.update({e["go_id"]:1}) for e in data if e["go_id"] in goIds]
            fullSet.append(record)
        else:
            fullSet.append(record)
    df = pd.DataFrame(fullSet)
    df.to_csv(f"{outputdir}/{datasetName}.csv", index=False, encoding="utf-8")


def runner(graph,featureFileName,outputdir="./cql/config/dataset"):
    # 'GO', 'INTERPRO', 'KEGG', 'Protein', 'REACTOME', 'index'
    config = json.load(open(featureFileName))
    index = config["index"]
    keys = list(filter(lambda x: x != "index", config.keys()))

    options = {
        "GO": getGO,
        "INTERPRO": getInterPro,
        "KEGG": getKeggPathWays,
        "Protein": createProteinMatrix,
        "REACTOME": getReactome
    }

    threads = [Thread(target=options[k],args=(graph,index,config[k],outputdir,),name=k) for k in keys]

    for t in threads:
        t.start()
    for t in threads:
        t.join()