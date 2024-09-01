Flight Delay Prediction using LSTM
Overview
This project focuses on predicting flight delays using Long Short-Term Memory (LSTM) models. The main objective is to estimate the arrival delay of flights based on historical flight data and various features related to airlines, origin-destination pairs, and tail numbers.

Dataset
The dataset consists of cleaned flight data, including information about flight times, delays, airline codes, tail numbers, origin and destination airports, and various other flight-related features. The data is processed to create additional features that are crucial for delay prediction.

Features
Common Features
CRSArrTimestamp: Scheduled arrival timestamp.
CRSDepTimestamp: Scheduled departure timestamp.
OD_pair: Combination of Origin and Destination airports.
tail_number_effect: Time since the last arrival of the same tail number.
RefDepDelay: Departure delay at the reference time.
CRSTime: Time in hours derived from the CRS timestamp.
flight_type: Indicates whether the flight is arriving or departing.
One-Hot Encoded Features
IATA_CODE_Reporting_Airline: One-hot encoding of airline codes.
Frequency Encoded Features
OD_pair: Frequency encoding of origin-destination pairs.
Model Architecture
The model is built using an LSTM (Long Short-Term Memory) neural network, which is well-suited for time series data. The LSTM model is used to capture the temporal dependencies in flight delay data. The architecture consists of the following components:

LSTM Layer: Captures temporal dependencies.
Dense Layer: Outputs the predicted delay.
Model Parameters
Input Shape: (n_timesteps, n_features) where n_timesteps is 1, and n_features is the number of input features.
Number of LSTM Units: 50.
Optimizer: Adam.
Loss Function: Mean Squared Error (MSE).
Methodology
Data Preprocessing
Timestamp Conversion: Flight times are converted into timestamps using convert_to_timedelta().
Feature Engineering: Additional features are created, such as tail_number_effect, RefDepDelay, and others.
Data Reduction: Airports with a low average daily flight count are filtered out to reduce data size.
Modeling Approaches
Two modeling approaches are used:

Approach 1: Airport-Specific Modeling
Predict delays by training a separate LSTM model for each airport.
Features include nearest arrival and departure times, flight type, and delay at reference times.
Approach 2: Tail Number-Specific Modeling
Predict delays by training a separate LSTM model for each tail number.
Features include tail_number_effect, delay at reference times, and other temporal features.
Model Training and Evaluation
The dataset is split into training and testing sets (80% train, 20% test).
The LSTM model is trained on the training set and evaluated on the test set using RMSE (Root Mean Squared Error) and MAE (Mean Absolute Error).
Models are trained for different reference times (15, 30, 60, 90 minutes).
Results
Performance metrics (RMSE, MAE) are calculated for both approaches and different reference times.
Predicted delays are compared against actual delays, and results are aggregated for visualization.
Visualization
F1 Plots: Visualize the predicted delays against the actual delays for selected airports.
Comparison of Reference Times: Analyze the impact of different reference times on prediction accuracy.
Dependencies
Python 3.x
Pandas
NumPy
TensorFlow
Scikit-learn
Matplotlib
tqdm
How to Run
Install the required packages:

bash
Copy code
pip install pandas numpy tensorflow scikit-learn matplotlib tqdm
Prepare the dataset: Ensure that the dataset is cleaned and preprocessed as per the data_cleaned DataFrame.

Run the model training: Execute the provided code to train the LSTM models for each approach.

Visualize the results: Use Matplotlib to visualize the predicted delays against actual delays.

Conclusion
This project demonstrates the use of LSTM models to predict flight delays with a focus on airport-specific and tail number-specific modeling approaches. The results indicate the effectiveness of these models in capturing temporal dependencies and predicting delays accurately.