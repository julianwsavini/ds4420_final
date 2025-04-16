# Rossmann Store Sales Prediction

## Overview
This project explores the application of machine learning models to predict future retail sales using the Rossmann store dataset. Two models were implemented: a Multi-Layer Perceptron (MLP) and a Seasonal Autoregressive Integrated Moving Average (SARIMA) model. The project demonstrates that even relatively simple ML models can achieve impressive results in sales forecasting, with the MLP model achieving an R-squared value of 0.97 for daily predictions and the SARIMA model achieving 0.66 for weekly predictions.

## Features
- **Data Processing**: Includes lag features, rolling statistics, exponentially weighted features, and cyclical encoding
- **MLP Model**: Implements a neural network with one hidden layer (80 nodes, ReLU activation)
- **SARIMA Model**: Optimizes parameters for weekly sales forecasting
- **Interactive Web App**: Allows users to select any Rossmann store and run the model on it
- **Future Work**: Plans to expand the MLP model to include data from multiple stores and incorporate additional features such as weather and promotions

## Authors
Kamil Pacana, Julian Savini, and Ved Rajesh (Northeastern University)
