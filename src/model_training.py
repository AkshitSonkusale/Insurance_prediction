# 1 load processed data
# 2 select 4 features
# 3 scale data
# 4 train model
# 5 save scaler and model

import pandas as pd
import pickle

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# load datasets
x_train = pd.read_csv("../data/processed/x_train.csv")
x_test = pd.read_csv("../data/processed/x_test.csv")
y_train = pd.read_csv("../data/processed/y_train.csv")
y_test = pd.read_csv("../data/processed/y_test.csv")

# keep only first 4 features
x_train = x_train.iloc[:, :4]
x_test = x_test.iloc[:, :4]

print("Training features shape:", x_train.shape)

# create scaler
scaler = StandardScaler()

# scale data
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# create model
model = LinearRegression()

# train model
model.fit(x_train_scaled, y_train)

# save scaler
with open("../artifacts/scaler.pkl","wb") as f:
    pickle.dump(scaler,f)

# save model
with open("../artifacts/model.pkl","wb") as f:
    pickle.dump(model,f)

print("Model and scaler saved successfully")