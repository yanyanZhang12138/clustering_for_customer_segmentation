import pandas as pd
import numpy as np
from sklearn.utils import shuffle

df_test = pd.read_csv('feature_matrix_1.csv')

def rand_sample(dataset, ratio):
    X_list = list(range(9))
    for i in X_list:
        X_test_shuffle = shuffle(df_test)
        X_list[i] = X_test_shuffle.sample(frac = ratio).to_numpy()
    return X_list

X_list = rand_sample(df_test, ratio = 0.1)
print(X_list[0])

