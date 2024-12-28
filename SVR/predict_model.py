import mlflow
logged_model = r'E:\Estudos\Alura\mlartifacts\385109406993758075\ff4a0d139fa246069fb7ad943354e670\artifacts\model'

# Load model as a PyFuncModel.
loaded_model = mlflow.pyfunc.load_model(logged_model)

# Predict on a Pandas DataFrame.
import pandas as pd
data = pd.read_csv('casas_X.csv')
predicted  = loaded_model.predict(pd.DataFrame(data))

data['predicted'] = predicted
data.to_csv('precos.csv')