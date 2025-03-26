# Rossmann Store Sales Prediction

## Overview
This project aims to forecast Rossmann store sales using both time series models and neural network approaches (MLP). We tackle the problem of predicting daily sales for multiple Rossmann drug stores across Europe, which has significant business implications for inventory management, staff scheduling, and financial planning.

## Dataset
We use the [Rossmann Store Sales dataset](https://www.kaggle.com/datasets/shahpranshu27/rossman-store-sales) from Kaggle, which includes historical sales data for 1,115 Rossmann stores. The dataset contains the following features:

* **Store** - a unique ID for each store
* **Sales** - the turnover for any given day (this is what we're predicting)
* **Customers** - the number of customers on a given day
* **Open** - an indicator for whether the store was open: 0 = closed, 1 = open
* **StateHoliday** - indicates a state holiday. Normally all stores, with few exceptions, are closed on state holidays. Note that all schools are closed on public holidays and weekends. a = public holiday, b = Easter holiday, c = Christmas, 0 = None
* **SchoolHoliday** - indicates if the (Store, Date) was affected by the closure of public schools
* **StoreType** - differentiates between 4 different store models: a, b, c, d
* **Assortment** - describes an assortment level: a = basic, b = extra, c = extended
* **CompetitionDistance** - distance in meters to the nearest competitor store
* **CompetitionOpenSince[Month/Year]** - gives the approximate year and month of the time the nearest competitor was opened
* **Promo** - indicates whether a store is running a promo on that day
* **Promo2** - Promo2 is a continuing and consecutive promotion for some stores: 0 = store is not participating, 1 = store is participating
* **Promo2Since[Year/Week]** - describes the year and calendar week when the store started participating in Promo2
* **PromoInterval** - describes the consecutive intervals Promo2 is started, naming the months the promotion is started anew. E.g. "Feb,May,Aug,Nov" means each round starts in February, May, August, November of any given year for that store

## Models
We implement and compare two main approaches:

### 1. Time Series Models
- **Bayesian Structural Time Series (BSTS)** - Captures trend, seasonality, and the effects of predictor variables
- **SARIMA (Seasonal ARIMA)** - Handles time-dependent structures in the data
- **Prophet** - Facebook's forecasting tool designed to handle multiple seasonality patterns

### 2. Neural Network Models
- **Multilayer Perceptron (MLP)** - A feed-forward neural network tailored for sales prediction
- **Temporal Fusion Transformer (Optional)** - A state-of-the-art architecture for interpretable time series forecasting

## Project Structure
rossmann-sales-prediction/
│
├── data/                       # Raw and processed data
│   ├── raw/                    # Original dataset files
│   └── processed/              # Cleaned and feature-engineered data
│
├── models/                     # Model implementations
│   ├── time_series/            # Time series model implementations
│   └── neural_networks/        # MLP and other neural network implementations
│
├── notebooks/                  # Jupyter notebooks for exploration and visualization
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_time_series_models.ipynb
│   └── 04_mlp_implementation.ipynb
│
├── src/                        # Source code
│   ├── data/                   # Data processing scripts
│   │   ├── make_dataset.py     # Data cleaning and transformation
│   │   └── feature_engineering.py
│   │
│   ├── models/                 # Model implementation code
│   │   ├── train_model.py      # Training procedures
│   │   ├── predict_model.py    # Making predictions with trained models
│   │   └── evaluate_model.py   # Model evaluation metrics
│   │
│   └── visualization/          # Visualization scripts
│       └── visualize.py
│
├── tests/                      # Test cases
│
├── requirements.txt            # Project dependencies
│
└── README.md                   # This file

## Results and Evaluation
We evaluate our models based on:
- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)
- MAPE (Mean Absolute Percentage Error)
- Feature importance analysis

Detailed results and visualizations can be found in the notebooks.

## References
- [Demand Forecasting in Supply Chain Management](https://paperswithcode.com/paper/demand-forecasting-in-supply-chain-management)
- Kaggle competition: [Rossmann Store Sales](https://www.kaggle.com/competitions/rossmann-store-sales)
- [Facebook Prophet documentation](https://facebook.github.io/prophet/)
- [Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting](https://arxiv.org/abs/1912.09363)

## Contributing
1. Fork the repository
2. Create your feature branch: `git checkout -b feature/amazing-feature`
3. Commit your changes: `git commit -m 'Add some amazing feature'`
4. Push to the branch: `git push origin feature/amazing-feature`
5. Open a Pull Request

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
- Rossmann for providing the dataset
- Kaggle for hosting the competition
