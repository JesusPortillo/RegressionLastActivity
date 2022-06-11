from email.policy import default
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model

scale = StandardScaler()
#read the data

df = pd.read_excel("Netflix_list.xlsx").dropna()
#normalization for type
condition_list = [
    (df["type"] == "Movie"),
    (df["type"] == "TV Show")
]
choice_list = [0,1]
#we add a new column with the type normalized
df["type_normalized"] = np.select(condition_list, choice_list, default = "Not_specified")

separate_duration_movies = df["duration"].str.split(expand=True)
df.insert(4,"durationInt", separate_duration_movies[0].astype(int))
print(df)
x,y = df["durationInt"], df["type_normalized"]
x,y = np.array(x).reshape(-1,1), np.array(y)
scaledX = scale.fit_transform(x)
train_X, train_y = scaledX[:1000], y[:1000]
test_x, test_y = scaledX[1000:1200], y[1000:1200]

plt.scatter(scaledX[:100], y[:100])
plt.show()
#this will work with a classification model due to it could be categorized
model = linear_model.LinearRegression().fit(train_X, train_y)

print(model.score(train_X, train_y))
print(model.score(test_x, test_y))
