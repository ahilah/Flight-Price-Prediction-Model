import pickle
import pandas as pd
# scale numerical features to a specified range
from sklearn.preprocessing import MinMaxScaler

# custom files
import columns

# set display options
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)

# file paths
data_path = "D:/programming/information-technologies-of-smart-systems/term-paper/data/final/new_data.csv"
param_dict_path = "D:/programming/information-technologies-of-smart-systems/term-paper/src/param_dict.pickle"
model_path = "D:/programming/information-technologies-of-smart-systems/term-paper/models/finalized_rf_model.sav"
prediction_result_path = "D:/programming/information-technologies-of-smart-systems/term-paper/data/prediction_results.csv"

# load dataset
try:
    ds = pd.read_csv(data_path)
    print('New data size: ', ds.shape)
except FileNotFoundError:
    print("Error: File not found!")
    exit()

# load parameter dictionary
try:
    with open(param_dict_path, 'rb') as f:
        param_dict = pickle.load(f)
        print("\nParam dict is loaded!")
except FileNotFoundError:
    print("Error: Parameter dictionary file not found!")
    exit()

# dropping the useless column 'Unnamed: 0'
ds = ds.drop('Unnamed: 0', axis=1)
print("column 'Unnamed: 0' is dropped")

# dropping the useless column 'flight'
ds = ds.drop('flight', axis=1)
print("column 'flight' is dropped")

# rename 'class' name to 'flight_class', because *class* is python reserved name
ds.rename(columns={'class': 'flight_class'}, inplace=True)
print("column 'class' is renamed to 'flight_class'")

# categorical encoding
for column in columns.cat_columns[0:]:
    ds[column] = ds[column].map(param_dict['map_dicts'][column])

# normalization
scaler = MinMaxScaler()
# fit the scaler to the train set, it will learn the parameters
X_tmp = ds[columns.X_columns]
# transform train and test sets
scaler = MinMaxScaler().fit_transform(X_tmp)
# list of column names in X_ymp
columns_to_replace = X_tmp.columns
# let's transform the returned NumPy arrays to dataframes for the rest of the dataset
X = pd.DataFrame(scaler, columns=X_tmp.columns)
# iterate over the columns to replace in df_encoded_MinMaxScaler
for column in columns_to_replace:
    ds[column] = X[column]

print("data is normalized")

# define target and features columns
X = ds[columns.X_columns]

# load the model and predict
try:
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
        print("Prediction model is loaded!")
except FileNotFoundError:
    print("Error: Model file not found!")
    exit()

try:
    y_pred = model.predict(X)
    ds['price_pred'] = model.predict(X)
    ds.to_csv(prediction_result_path, index=False)
    print("Predicted prices are in file", prediction_result_path)
except Exception as e:
    print("Error occurred while predicting:", e)
