import pickle
import numpy as np
import os

class Insurance_prediction:

    def __init__(self):

        base_dir = os.path.dirname(os.path.dirname(__file__))

        scaler_path = os.path.join(base_dir, "artifacts", "scaler.pkl")
        model_path = os.path.join(base_dir, "artifacts", "model.pkl")

        with open(scaler_path, "rb") as f:
            self.scaler = pickle.load(f)

        with open(model_path, "rb") as f:
            self.model = pickle.load(f)

   def prediction(self, Age, Annual_Income_LPA, Policy_Term_Years, Sum_Assured_Lakhs, Annual_Premium_Thousands):

    input_data = np.array([[Age, Annual_Income_LPA, Policy_Term_Years, Sum_Assured_Lakhs, Annual_Premium_Thousands]])

    scaled_input = self.scaler.transform(input_data)

    result = self.model.predict(scaled_input)

    return result[0]
