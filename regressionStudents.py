import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model

scale = StandardScaler()
#read the data

df = pd.read_csv("student_data.csv")
print(df.corr())
x = df[["G1","G2"]]
y = df[["G3"]]
#scale data

scaledX = scale.fit_transform(x)

#segment the data for training and testing
train_x = scaledX[:316]
train_y = y[:316]

test_x = scaledX[316:]
test_y = y[316:]
#model
model = linear_model.LinearRegression().fit(train_x, train_y)
#prove with training data
print(model.score(train_x,train_y))
#prove with testing data
print(model.score(test_x, test_y))
