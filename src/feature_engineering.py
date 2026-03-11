#1 load the training and testing data 
#2 scale the training data 
#3 save scaled data in processed folder

#1
from sklearn.preprocessing import StandardScaler # import this to scale the data 
from data_preprocessing import load_and_split_data
import pandas as pd
import pickle 

x_train,x_test,y_train,y_test=load_and_split_data()

scaler=StandardScaler()

x_train_scaled=scaler.fit_transform(x_train)
x_test_scaled=scaler.fit_transform(x_test)

pd.DataFrame(x_train_scaled).to_csv("../data/processed/x_train.csv",index=False)
pd.DataFrame(x_test_scaled).to_csv("../data/processed/x_test.csv",index=False)
pd.DataFrame(y_train).to_csv("../data/processed/y_train.csv",index=False)
pd.DataFrame(y_test).to_csv("../data/processed/y_test.csv",index=False)

with open("../artifacts/scaler.pkl","wb") as f:
    pickle.dump(scaler,f)