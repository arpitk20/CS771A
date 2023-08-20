import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import pickle as pkl


def myfit(df_train):
    df_feat = df_train.drop(
    	["OZONE", "NO2", "Time"], axis="columns")
    target = df_train[['OZONE','NO2']]
    model = DecisionTreeRegressor(criterion='squared_error',random_state=42)
    model.fit(df_feat, target)
    with open(f"model", "wb") as outfile:
	    pkl.dump(model, outfile, protocol=pkl.HIGHEST_PROTOCOL)


df_train = pd.read_csv("train.csv")
df_test = pd.read_csv("dummy_test.csv")
myfit(df_train)
# my_predict(df_test)
