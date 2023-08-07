import os 

import numpy as np


CURRENT_DIR = os.path.dirname(__file__)
iris_relative_path = "../data/iris/iris.data"
iris_absolute_path = os.path.join(CURRENT_DIR, iris_relative_path)

with open(iris_absolute_path) as f:
    lines = [i[:-1] for i in f.readlines()]
    
n = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]
x = [n.index(i.split(",")[-1]) for i in lines if i != ""]
x = np.array(x, dtype="uint8")

y = [[float(j) for j in i.split(",")[:-1]] for i in lines if i != ""]
y = np.array(y)

i = np.argsort(np.random.random(x.shape[0]))
x = x[i]
y = y[i]

np.save(os.path.join(CURRENT_DIR, "../data/prepared/iris_features.npy"), y)
np.save(os.path.join(CURRENT_DIR, "../data/prepared/iris_labels.npy"), x)