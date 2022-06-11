import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model

scale = StandardScaler()
#read the data
df = pd.read_csv("cars2.csv")
x = df[["Volume","Weight"]]
y = df[["CO2"]]
#scale data
scaledX = scale.fit_transform(x)
#segment the data for training and testing
train_x = scaledX[:24]
train_y = y[:24]

test_x = scaledX[24:]
test_y = y[24:]
#model
model = linear_model.LinearRegression().fit(train_x, train_y)
#prove with training data
print(model.score(train_x,train_y))
#prove with testing data
print(model.score(test_x, test_y))
#-----------------------------------------------------------------------------#

