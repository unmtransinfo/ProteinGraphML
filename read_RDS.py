import pyreadr
import pickle
import numpy as np
data = pyreadr.read_r('104300.rds')
#print (data)
labels = {}
labels[True] = set(np.where(data[None]['Y']=='pos')[0])
labels[False] = set(np.where(data[None]['Y']=='neg')[0])

with open('104300.pkl', 'wb') as handle:
    pickle.dump(labels, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('104300.pkl', 'rb') as handle:
    unserialized_data = pickle.load(handle)

print (unserialized_data)
