#STEP 1 : load raw data 
#STEP 2 : identify x and y ( input and output feature )
#STEP 3 : Split data into training and testing data 

# 1
import pandas as pd
from sklearn.model_selection import train_test_split

data=pd.read_csv("../data/raw/insurance_data.csv")

# 2
def load_and_split_data():
    x=data[['Age','Annual_Income_LPA','Policy_Term_Years','Sum_Assured_Lakhs']]
    y=data['Annual_Premium_Thousands']
    print(x)
    print(y)

# 3
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
    return x_train,x_test,y_train,y_test
