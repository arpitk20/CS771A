import numpy as np
import pandas as pd
import pickle as pkl


def my_predict(df):
	with open("model", "rb") as file:
		model = pkl.load(file)

	df_feat = df.drop(["Time"], axis=1)
	output = model.predict(df_feat)
	o3 = output[:, 0]
	no2 = output[:, 1]
	return (o3, no2)
