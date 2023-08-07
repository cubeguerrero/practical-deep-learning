import os

import numpy as np
import matplotlib.pyplot as plt


CURRENT_DIR = os.path.dirname(__file__)
data_rel_path = "../data/breast_cancer/wdbc.data"
data_abs_path = os.path.join(CURRENT_DIR, data_rel_path)

with open(data_abs_path) as f:
    lines = [i[:-1] for i in f.readlines() if i != ""]
    

n = ["B", "M"]
x = np.array([n.index(i.split(",")[1]) for i in lines], dtype="uint8")
y = np.array([[float(j) for j in i.split(",")[2:]] for i in lines])
i = np.argsort(np.random.random(x.shape[0]))
x = x[i]
y = y[i]
z = (y - y.mean(axis=0)) / y.std(axis=0)

np.save(os.path.join(CURRENT_DIR, "../data/prepared/bc_features.npy"), y)
np.save(os.path.join(CURRENT_DIR, "../data/prepared/bc_features_standard.npy"), z)
np.save(os.path.join(CURRENT_DIR, "../data/prepared/bc_labels.npy"), x)
plt.boxplot(z)
plt.show()