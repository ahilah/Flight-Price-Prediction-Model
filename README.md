# Flight Ticket Price Prediction Model

## [Dataset](https://www.kaggle.com/datasets/shubhambathwal/flight-price-prediction/data?select=Clean_Dataset.csv) description features
1) Airline. The name of the airline company is stored in the airline column. It is a categorical feature having 6 different airlines.
2) Flight. Flight stores information regarding the plane's flight code. It is a categorical feature.
3) Source City. City from which the flight takes off. It is a categorical feature having 6 unique cities.
4) Departure Time. This is a derived categorical feature obtained created by grouping time periods into bins. It stores information about the departure time and have 6 unique time labels.
5) Stops. A categorical feature with 3 distinct values that stores the number of stops between the source and destination cities.
6) Arrival Time. This is a derived categorical feature created by grouping time intervals into bins. It has six distinct time labels and keeps information about the arrival time.
7) Destination City. City where the flight will land. It is a categorical feature having 6 unique cities.
8) Class. A categorical feature that contains information on seat class; it has two distinct values: Business and Economy.
9) Duration. A continuous feature that displays the overall amount of time it takes to travel between cities in hours.
10) Days Left. This is a derived characteristic that is calculated by subtracting the trip date by the booking date.
11) Price. Target variable stores information of the ticket price.

## Request
This model aims to assist an insurance company [EaseMyTrip](https://www.easemytrip.com/) in forecasting the prices for tickets in flight. The primary goal is to provide accurate and reliable predictions to help the company optimize pricing strategies, enhance customer satisfaction, and improve operational efficiency. By utilizing machine learning algorithms, the model can analyze historical data and identify patterns to predict future ticket prices.


### Requirements

    python 3.7

    numpy==1.17.3
    pandas==1.1.5
    sklearn==1.0.0


### Running:

    To run the demo, execute:
        python predict.py 

    After running the script in the folder './data/' will be generated <prediction_results.csv> 
    The file has 'price_pred' column with the predicted result value.

    The input is expected csv file in the folder './data/final/' with a name <new_data.csv>. The file should have all features columns. 

### Training a Model:

    Before you run the training script for the first time, you must create dataset. The file <train_data.csv> should contain all features columns and target for prediction of Price.
    After running the script the "param_dict.pickle" and "finalized_/rf/dtree/xgb_model.saw" will be created.
    
    Run the training script:
        python train.py

    The model achieves a score of 98%, with an error of approximately 2%.
    Note that there is currently no fraud check implemented.


#### Scripts
The project includes several Python scripts:
- predict.py. Implements the prediction model using machine learning techniques. It loads the model and predicts flights ticket prices based on input features.
- model-training.py. Develops and trains the machine learning model using the best 3 models: RandomForestRegressor, DecisionTreeRegressor & XGBRegressor. Contains regression evaluation metrics and model training process using various algorithms such as Linear Regression, Ridge Regression, Lasso Regression, K-Nearest Neighbors, Random Forest, XGBoost, Decision Tree, Gradient Boosting, and Extra Tree.
- feature-engineering.py. Performs feature engineering tasks such as categorical encoding.
- split-data. Splits the data to Train/Test sets.

#### Model Evaluation Metrics
The model performance is evaluated using the following metrics:
1. Score.
2. Mean Squared Error (MSE).
3. Mean Absolute Error (MAE).
4. **R2 Score** is the most important one.
The results are saved in a CSV file named like 'random-forest-metrics.csv' in the folder './models/'.

#### Model Selection
The notebook model-training.ipynb iterates through various regression algorithms and selects the best-performing model based on the root mean squared error (RMSE) on the test data. The most suited models are saved using pickle for future use: RandomForestRegressor, DecisionTreeRegressor and XGBRegressor.

#### Feature Importance
The notebook also analyzes the feature importance using the 3 selected models (RandomForestRegressor, DecisionTreeRegressor & XGBRegressor). All the features are visualized using a bar plot.