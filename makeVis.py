

# SAVED DATA 
# WHERE TO SAVE 
# TYPE, FIG OR GRAPH 
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter

from ProteinGraphML.Analysis.featureLabel import convertLabels
from ProteinGraphML.Analysis import Visualize

from ProteinGraphML.DataAdapter import OlegDB,selectAsDF
from ProteinGraphML.GraphTools import ProteinDiseaseAssociationGraph

import pickle
## we construct a base map of protein to disease just by creating the ProteinDiseaseAs

#print('hehe')
def featureVisualize(features,AUC,TITLE,count=20): 
    
	plt.rcParams.update({'font.size': 15,'lines.linewidth': 1000}) #   axes.labelweight': 'bold'})
	FILETITLE = TITLE
	TITLE = TITLE + "-AUC: "+str(AUC)  

	df = pd.DataFrame(features.most_common(count), columns=['feature', 'gain'])
	plt.figure()
	df['gain'] = (df['gain']/sum(df['gain'][:count]))
	#df['feature'] = df['feature'].map(processFeature)
	r = df.head(count).plot( kind='barh',title=TITLE, x='feature', y='gain',color='tomato', legend=False, figsize=(10, 12))
	r.set_xlabel('Importance')
	r.set_ylabel('Features')
	r.invert_yaxis()

	r.figure.savefig(FILETITLE+'.png',bbox_inches='tight')


dbAdapter = OlegDB()

def load_obj(name):
    with open('results/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)

importance = load_obj('firsty')

#loadedObject
labelMap = convertLabels(importance.keys(),dbAdapter,selectAsDF,type='plot')

print('HERE ARE LABELS')

#importance = {'hsa01100': 0.31735258141642814, 'hsa04740': 0.2208299216149202, 'hsa05100': 0.1847905733996812, 'hsa04930': 0.10625980494746863, 'hsa04514': 0.047493659101048136, 'hsa04114': 0.03542724660274679, 'hsa04810': 0.03365848585388666, 'hsa04144': 0.030556051003490892}#{"MP_0000180":34,343:1.0,30001:0.3}
#labelMap = convertLabels(importance.keys(),dbAdapter,selectAsDF,type='plot')
#for value[key] in importance.values():

newSet = {}
for key in importance.keys():
	newSet[labelMap[key]] = importance[key]


AUC = 0.9
#print(newSet,labelMap)

if False:
	currentGraph = ProteinDiseaseAssociationGraph.load("newCURRENT_GRAPH")

# for the graph, we need the original importance
	for key in importance.keys():
		Visualize((key,importance[key]),currentGraph.graph,"MP_0000180",dbAdapter=dbAdapter) #g,currentGraph.graph,Disease)
		break

print('STARTING FEAT VIS')

featureVisualize(Counter(newSet),AUC,"AAA")
#Visualize




#convertLabels([343,30001],dbAdapter,selectAsDF,type="han")
