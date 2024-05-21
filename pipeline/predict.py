import pickle
import pandas as pd

# custom files
import columns

# set display options
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)

# file paths
data_path = "D:/programming/information-technologies-of-smart-systems/calculation-and-graphic work/personal-medical-insurance-cost-prediction/data/final/new_data.csv"
param_dict_path = "D:/programming/information-technologies-of-smart-systems/calculation-and-graphic work/personal-medical-insurance-cost-prediction/src/param_dict.pickle"
model_path = "D://programming//information-technologies-of-smart-systems//calculation-and-graphic work//personal-medical-insurance-cost-prediction//models//finalized_model.sav"
prediction_result_path = "D:/programming/information-technologies-of-smart-systems/calculation-and-graphic work/personal-medical-insurance-cost-prediction/data/prediction_results.csv"

# load dataset
try:
    ds = pd.read_csv(data_path)
    print('New data size:', ds.shape)
except FileNotFoundError:
    print("Error: File not found!")
    exit()

# load parameter dictionary
try:
    with open(param_dict_path, 'rb') as f:
        param_dict = pickle.load(f)
except FileNotFoundError:
    print("Error: Parameter dictionary file not found!")
    exit()

# categorical encoding
for column in columns.cat_columns[0:]:
    ds[column] = ds[column].map(param_dict['map_dicts'][column])

# define target and features columns
X = ds[columns.X_columns]

# load the model and predict
try:
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
except FileNotFoundError:
    print("Error: Model file not found!")
    exit()

try:
    y_pred = model.predict(X)
    ds['charges_pred'] = model.predict(X)
    ds.to_csv(prediction_result_path, index=False)
except Exception as e:
    print("Error occurred while predicting:", e)
