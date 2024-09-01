# Flight Delay Prediction using LSTM

## Overview

This project focuses on predicting flight delays using Long Short-Term Memory (LSTM) models. The main objective is to estimate the arrival delay of flights due to late aircraft delay and carrier delay based on historical flight data and various features related to airlines, origin-destination pairs, and tail numbers.

## Dataset

The dataset consists of flight data, including information about flight times, delays, airline codes, tail numbers, origin and destination airports, and various other flight-related features. The data is processed to create additional features that are crucial for delay prediction.

## Methodology
Carefully engineered features that capture the complex dynamics of flight operations, delays, and interdependencies are used.

### Data Preprocessing
1. **Data Cleaning**: Missing values in key columns are addressed through calculated imputation strategies, ensuring accurate delay prediction.
2. **Feature Engineering**: Temporal features, OD pair frequency encoding, and airline-specific delays are engineered to capture the operational dynamics of flights.
3. **Data Filtering**: The dataset is filtered to focus on high-traffic airports, improving the model's ability to generalize to critical nodes in the air traffic network.

### Modeling Approaches
Two primary modeling approaches are used in this project, each leveraging different feature sets to predict flight delays:

### Approach 1: Airport-Centric Modeling

In this approach, the model focuses on predicting delays based on airport-specific dynamics. The model is trained using **FS-1**, the airport-centric feature set, which captures factors such as flight congestion, runway scheduling, and the interaction between arriving and departing flights. The approach involves training a separate LSTM model for each airport, treating the airport as the focal point of operations.

### Approach 2: Aircraft-Centric Modeling

This approach targets delays by focusing on the operational patterns of individual aircraft, captured through **FS-2**, the aircraft-centric feature set. The model leverages tail number-specific features to predict delays that may propagate across different airports due to the rotation of aircraft. A separate LSTM model is trained for each aircraft, using the tail number as the key identifier.

### Common Features for Both Approaches

Certain features are fundamental to understanding flight delays and are used in both the airport-centric and aircraft-centric approaches:

- **OD_pair**: Origin-Destination (OD) pair effects are captured through frequency encoding. This feature reflects the frequency of flights between two airports, which can influence the likelihood of delays due to congestion or operational bottlenecks.
- **Airline Effect (One-Hot Encoding)**: Airlines are one-hot encoded to capture airline-specific delay patterns that may arise from internal operations or other carrier-specific factors.
- **Temporal Features (DayOfWeek, Month)**: These features extract the day of the week and month from the flight date, capturing seasonal and weekly patterns that may impact flight delays.

### Feature Set 1: Airport-Centric Flight Dynamics

This set of features is designed to capture the dynamics specific to airport operations, such as congestion, runway scheduling, and the interdependencies between arriving and departing flights. The model treats each airport as a central point of operation:

- **Flight Type (flight_type)**: A binary feature indicating whether the flight is arriving at (1) or departing (0) from the selected airport.
- **Nearest Arrival/Departure Before and After**: These features represent the time (in minutes) between the target flight and the nearest previous or next arriving/departing flight. This helps to understand how closely spaced the flights are at the arrival airport, which could affect congestion and delays.
- **Departure Delay Effect (RefDepDelay)**: This feature captures the delay at the reference time (`RefAt`), which is determined by subtracting a predefined number of minutes (15, 30, 60, 90) from the scheduled arrival timestamp. The delay is calculated as the difference (in minutes) between the actual departure and the scheduled departure time at `RefTime`. If the flight has not yet departed at `RefAt`, the value reflects the difference between `RefAt` and the scheduled departure time.
- **Scheduled Time (CRSTime)**: This feature converts the scheduled arrival or departure time into a fractional hour-minute format, making it easier for the model to capture temporal patterns.

This feature set, referred to as **FS-1**, is specifically applied to flights operating at five major airports due to computational constraints. It can be extended to other airports by removing the airport filter.

### Feature Set 2: Aircraft-Centric Delay Propagation

This set of features focuses on the operational patterns of individual aircraft, under the assumption that delays may propagate across different airports due to aircraft rotation schedules. The model treats each aircraft (identified by its tail number) as the central point of analysis:

- **Tail Number Reference Effect (tail_number_REF_effect)**: This feature reflects the impact of prior delays on the same tail number on the timing of future flights. If the previous flight’s actual arrival time (using the same aircraft) is after the reference time (`RefAt`), the delay effect is calculated as the difference between `RefAt` and the scheduled arrival time. Otherwise, the delay is calculated as the difference between the scheduled arrival time and the actual arrival time.
- **Scheduled Tail Number Effect (tail_number_effect)**: This feature captures the time difference (in minutes) between the scheduled departure of the current flight and the scheduled arrival time of the previous flight with the same tail number.
- **Scheduled Time (CRSTime)**: Similar to FS-1, this feature stores the aircraft’s scheduled departure time at the reference airport, converted into a fractional hour-minute format.

The features used in this approach are collectively referred to as **FS-2**. Although the current implementation is limited to five aircraft for demonstration purposes, this method can be extended to all aircraft in the dataset, given sufficient computational resources.


## Model Architecture

The model is built using an LSTM (Long Short-Term Memory) neural network, which is well-suited for time series data. The LSTM model is used to capture the temporal dependencies in flight delay data. The architecture consists of the following components:
- **LSTM Layer**: Captures temporal dependencies.
- **Dense Layer**: Outputs the predicted delay.

### Model Parameters
- **Input Shape**: `(n_timesteps, n_features)` where `n_timesteps` is 1, and `n_features` is the number of input features.
- **Number of LSTM Units**: 50.
- **Optimizer**: Adam.
- **Loss Function**: Mean Squared Error (MSE).


### Model Training and Evaluation
- The dataset is split into training and testing sets (80% train, 20% test).
- The LSTM model is trained on the training set and evaluated on the test set using RMSE (Root Mean Squared Error) and MAE (Mean Absolute Error).
- Models are trained for different reference times (15, 30, 60, 90 minutes).

### Results
- Performance metrics (RMSE, MAE) are calculated for both approaches and different reference times.
- Predicted delays are compared against actual delays, and results are aggregated for visualization.

## Visualization

- **F1 Plots**,**F2 Plots** : Visualize the predicted delays against the actual delays for selected airports.
- **Comparison of Reference Times**: Analyze the impact of different reference times on prediction accuracy.

## Dependencies

- Python 3.x
- Pandas
- NumPy
- TensorFlow
- Scikit-learn
- Matplotlib
- tqdm

## How to Run

1. **Install the required packages**:
    ```bash
    pip install pandas numpy tensorflow scikit-learn matplotlib tqdm
    ```

2. **Prepare the dataset**: Download the datasets for flight data (BTS website).

3. **Run the model training and Evaluation**: Execute the provided code to train the LSTM models for each approach.

## Conclusion

This project demonstrates the use of LSTM models to predict flight delays with a focus on airport-specific and tail number-specific modeling approaches. The results indicate the effectiveness of these models in capturing temporal dependencies and predicting delays accurately.
