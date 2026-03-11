import pickle
import numpy as np

class Insurance_prediction:

    def __init__(self):
        with open("artifacts/scaler.pkl","rb") as f:
            self.scaler = pickle.load(f)

        with open("artifacts/model.pkl","rb") as f:
            self.model = pickle.load(f)

    def prediction(self, Age, Annual_Income_LPA, Policy_Term_Years, Sum_Assured_Lakhs):

        input = np.array([[Age, Annual_Income_LPA, Policy_Term_Years, Sum_Assured_Lakhs]])

        scaled_input = self.scaler.transform(input)

        result = self.model.predict(scaled_input)

        return result[0]
<<<<<<< HEAD
=======
    
        print(self.scaler.n_features_in_)
>>>>>>> 23c26cd (updated model training)
