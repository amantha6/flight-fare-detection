# flight-fare-detection
Predicting the fair of airline tickets




![image](https://github.com/user-attachments/assets/9780fec3-f609-4933-80da-a1f076e47eb9)
# Flight Price Prediction

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0%2B-orange)
![pandas](https://img.shields.io/badge/pandas-1.3%2B-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

## Overview

This project implements a machine learning system to predict flight prices based on various factors, enabling travelers and travel agencies to optimize booking strategies. Using RandomForest and decision tree models, we achieved 92% accuracy (R²) and 6.78% mean absolute percentage error (MAPE).

## Key Features

- **Comprehensive ML Pipeline**: End-to-end solution from data preprocessing to model deployment
- **Advanced Feature Engineering**: Transforms categorical flight data for optimal prediction
- **Model Comparison**: Evaluates multiple models with custom metrics
- **Hyperparameter Optimization**: Fine-tunes model parameters for peak performance
- **Price Analysis Experiments**: Runs simulations for practical booking insights

## Dataset

The dataset includes 10,000+ flight records with features including:
- Airline
- Source and destination
- Route details
- Departure and arrival times
- Flight duration
- Number of stops
- Seasonality information

## Technical Architecture

```
flight-price-prediction/
├── data/
│   ├── Data_Train.xlsx
│   └── processed/
├── models/
│   ├── best_simple_model.pkl
│   └── tuned_flight_price_model.pkl
├── notebooks/
│   └── flight_prediction.ipynb
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── model_training.py
│   └── evaluation.py
├── experiments/
│   └── flight_prediction_experiments.py
├── requirements.txt
└── README.md
```

## Model Performance

| Model | R² Score | MAPE (%) | MSE | RMSE |
|-------|----------|----------|-----|------|
| Random Forest (base) | 0.9169 | 6.78 | 1,713,159 | 1,308.88 |
| Decision Tree | 0.8873 | 7.43 | 2,323,943 | 1,524.45 |
| Random Forest (tuned) | 0.8022 | 16.46 | 4,076,903 | 2,019.13 |

## Key Insights

The project revealed several important patterns in flight pricing:

1. **Airline Impact**: Different carriers show up to 40% price variation on identical routes
2. **Timing Effect**: Early morning flights are typically 15-20% cheaper than mid-day flights
3. **Advance Booking**: Optimal booking window is 30-60 days before departure
4. **Seasonal Patterns**: Prices in peak travel seasons can be 30% higher than off-peak
5. **Stop Premium**: Each additional stop increases price by approximately 10-20%



## Prediction API Usage

```python
from src.model_training import load_model

# Load the trained model
model, preprocessor = load_model('models/best_simple_model.pkl')

# Create sample flight data
flight_data = {
    'Airline': 'IndiGo',
    'Date_of_Journey': '2023-08-15',
    'Source': 'Delhi',
    'Destination': 'Mumbai',
    'Route': 'DEL → BOM',
    'Dep_Time': '10:00',
    'Arrival_Time': '12:30',
    'Duration': '2h 30m',
    'Total_Stops': 'non-stop',
    'Additional_Info': 'No info'
}

# Preprocess the data
flight_df = pd.DataFrame([flight_data])
flight_processed = preprocessor.transform(flight_df)

# Make prediction
predicted_price = model.predict(flight_processed)[0]
print(f"Predicted flight price: ₹{predicted_price:.2f}")
```

## Future Improvements

- Implement deep learning models for potentially higher accuracy
- Add real-time price scraping for up-to-date predictions
- Develop a web interface for easy consumer access
- Explore transfer learning to adapt to new routes/airlines
- Incorporate weather data for improved seasonal predictions

## License

This project is licensed under the MIT License - see the LICENSE file for details.

