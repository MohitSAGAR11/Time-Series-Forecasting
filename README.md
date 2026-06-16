# 🌡️ Univariate Time Series Forecasting on Jena Climate Dataset

An empirical benchmark comparing Simple Moving Average (SMA), Vanilla RNN, and LSTM architectures on the task of **1-step-ahead univariate temperature prediction** using the full **Jena Climate Dataset**.

---

## 📋 Table of Contents
1. [Project Overview](#-project-overview)
2. [Objectives & Success Criteria](#-objectives--success-criteria)
3. [Dataset Description](#-dataset-description)
4. [End-to-End Pipeline](#-end-to-end-pipeline)
5. [Model Architecture](#-model-architecture)
6. [Evaluation Metrics & Results](#-evaluation-metrics--results)
7. [Results & Insights](#-results--insights)
8. [Challenges & Limitations](#-challenges--limitations)
9. [Future Improvements](#-future-improvements)
10. [Project Structure](#-project-structure)
11. [Installation Guide](#-installation-guide)
12. [Usage Instructions](#-usage-instructions)
13. [Conclusion](#-conclusion)

---

## 🔍 Project Overview

This project benchmarks sequence models on the task of forecasting the next temperature measurement (10 minutes into the future) given a window of historical temperatures. Using the Jena Climate Dataset, we analyze the performance of:
1. **Simple Moving Average (SMA)**: A non-parametric baseline.
2. **Vanilla RNN**: A simple recurrent neural network.
3. **LSTM**: A gated recurrent network.

We evaluate how varying the **lookback window** (historical context length) affects prediction accuracy across 24, 72, 144, and 288 steps (corresponding to 4 hours, 12 hours, 24 hours, and 48 hours respectively).

---

## 🎯 Objectives & Success Criteria

- **Primary Goal**: Construct an end-to-end sequence learning pipeline in PyTorch to predict univariate temperature ($T$).
- **Research Question**: *How does the length of historical context (lookback window) affect forecast accuracy, and does it expose architectural limitations in recurrent models?*
- **Success Criteria**: Outperform statistical baselines and minimize Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE) on the test split.

---

## 🗂️ Dataset Description

- **Source**: Jena Climate Dataset (Max Planck Institute for Biogeochemistry), containing 7 years of meteorological measurements.
- **Size**: 420,551 observations at 10-minute intervals (spanning 2009 to 2016).
- **Target**: `T (degC)` (temperature) extracted and renamed to `temperature`.
- **Features**: Univariate input (only past values of the target variable are used).
- **Characteristics**: 
  - Strong daily (diurnal) and annual seasonality.
  - High short-term autocorrelation (temperature changes slowly step-to-step).
  - Wind velocity variables (`wv (m/s)` and `max. wv (m/s)`) contain sensor anomaly sentinel values of `-9999.0` (noted in EDA, but not impacting our univariate temperature pipeline).

---

## 🔄 End-to-End Pipeline

### 1. Data Loading & Indexing
- **What**: Loads `jena_climate_2009_2016.csv` using `kagglehub`. Parses the `Date Time` column, sets it as the index, and sorts it.
- **Why**: Ensures chronological ordering necessary for time-series splits.
- **Impact**: Provides a cleaned, sorted DataFrame of 420,551 rows.

### 2. Exploratory Data Analysis (EDA)
- **What**: Visualizes the full 7-year temperature timeline and zooms into July 2013 to observe diurnal cycles. Also plots humidity and pressure.
- **Why**: Verifies data integrity and confirms seasonality to guide lookback choices.
- **Impact**: Confirms strong diurnal cycles, showing that a 144-step window (24 hours) is a natural physical boundary.

### 3. Chronological Train-Val-Test Splitting
- **What**: Splits the target data into Train (70%), Validation (15%), and Test (15%) splits chronologically.
  - Train: Index 0 to 294,385
  - Val: Index 294,385 to 357,467
  - Test: Index 357,467 to 420,551
- **Why**: Shuffling would cause future-to-past data leakage.
- **Impact**: Sets up a leakage-free split for modeling.

### 4. MinMaxScaler Scaling
- **What**: Normalizes the temperature values to a $[0, 1]$ range using scikit-learn's `MinMaxScaler`. The scaler is fit *only* on the training data.
- **Why**: Neural networks converge faster and more stably on scaled inputs. Fitting on training data only prevents range leakage.
- **Impact**: Outputs scaled 1D arrays for input generation.

### 5. Sequence Generation (Sliding Window)
- **What**: Implements a custom PyTorch `TimeSeriesDataset` that uses a sliding window to generate 3D input tensors of shape `(batch_size, lookback_steps, 1)` and targets of shape `(batch_size, 1)`.
- **Why**: Prepares data in the 3D structure expected by PyTorch recurrent layers.
- **Impact**: Generates DataLoader objects with a default batch size of 64.

### 6. Training & Evaluation
- **What**: Train neural models using MSE loss and the Adam optimizer. Predictions are inverse-transformed back to the original Celsius scale before calculating MAE and RMSE.
- **Why**: Evaluation in Celsius provides a direct, interpretable measure of performance.
- **Impact**: Generates evaluation metrics on the test set.

---

## 🧠 Model Architecture

### 1. Simple Moving Average (SMA) Baseline
- **Rationale**: Predicts the arithmetic mean of the lookback window.
- **Pros**: Zero parameters, no training, fast baseline.
- **Cons**: Severe lag; averages out recent trends and performs poorly on long windows.

### 2. Vanilla RNN
- **Layer-by-Layer**:
  - Input: `(batch_size, lookback, 1)`
  - Recurrent layer: `nn.RNN(input_size=1, hidden_size=64, num_layers=1, batch_first=True)` with tanh activation
  - Output head: `nn.Linear(64, 1)` applied to the final hidden state $h_T$.
- **Total Parameters**: 4,353
- **Pros**: Lightweight, fast training, highly responsive to recent changes.
- **Cons**: Susceptible to vanishing gradients over long sequences.

### 3. LSTM (Long Short-Term Memory)
- **Layer-by-Layer**:
  - Input: `(batch_size, lookback, 1)`
  - Recurrent layers: 2-layer stacked `nn.LSTM(input_size=1, hidden_size=64, num_layers=2, batch_first=True, dropout=0.2)`
  - Regularization: `nn.Dropout(0.2)`
  - Output head: `nn.Linear(64, 32)` -> `nn.ReLU()` -> `nn.Linear(32, 1)`
- **Total Parameters**: 51,489
- **Pros**: Gated structure captures long-range dependencies and mitigates vanishing gradients.
- **Cons**: Prone to overfitting/instability on simple 1D inputs; much larger parameter size.

---

## 📊 Evaluation Metrics & Results

All metrics are evaluated on the Test set and reported in original Celsius units (°C).

### Test Metrics (Lookback = 24 steps / 4 hours)
- **Simple Moving Average**: MAE = 1.2014 °C, RMSE = 1.6382 °C
- **Vanilla RNN (50 epochs, patience 7)**: MAE = **0.1250 °C**, RMSE = **0.1954 °C**
- **LSTM (50 epochs, patience 7)**: MAE = 0.2981 °C, RMSE = 0.4257 °C

### Lookback Sweep Summary Table (MAE in °C)
*Note: Trained for 10 epochs with patience 3 during the sweep loop.*

| Lookback Steps (Duration) | Simple Moving Average | Vanilla RNN | LSTM |
|:---:|:---:|:---:|:---:|
| **24 (4 hours)** | 1.2014 | 0.1327 | 0.5213 |
| **72 (12 hours)** | 2.5324 | 0.1368 | 0.2306 |
| **144 (24 hours)** | 2.4668 | **0.1246** | 0.2933 |
| **288 (48 hours)** | 2.8050 | 0.1249 | 0.2745 |

---

## 📈 Results & Insights

1. **Vanilla RNN Superiority**: The Vanilla RNN outperformed the LSTM across *all* lookback windows. Its test MAE remained exceptionally low (~0.125 °C), whereas the LSTM's error was significantly higher (0.23 °C to 0.52 °C).
2. **Optimal Lookback**: The Vanilla RNN achieved its lowest error (MAE = 0.1246 °C) at 144 steps (24 hours), which aligns with the physical diurnal cycle.
3. **SMA Degradation**: As lookback increases, SMA error rises drastically (from 1.20 °C to 2.80 °C) because it averages out the highly correlated recent values.

---

## ⚠️ Challenges & Limitations (CRITICAL CALLOUTS)

> [!WARNING]
> **1. Narrative-Data Contradiction**
> The notebook text claims "LSTM consistently outperforms both baselines" and "RNN underperforms LSTM at long lookbacks." However, the actual code outputs clearly contradict this: Vanilla RNN (MAE ~0.125 °C) severely outperforms the LSTM (MAE ~0.27 °C) in every lookback sweep.
> 
> **2. Early Stopping Anomaly**
> The LSTM training loop experienced early stopping at Epoch 8 because validation loss rose after Epoch 1 (Epoch 1: 0.000150, Epoch 5: 0.000427). Reverting to Epoch 1 weights left the model under-trained, explaining its poor test performance.
> 
> **3. Unfair Model Comparison**
> The architectural comparison is uncontrolled. The Vanilla RNN is a single-layer network with a direct linear head, while the LSTM has 2 layers, 0.2 dropout, a 2-layer MLP head, and ReLU. This extra complexity causes training instability on simple univariate inputs.
> 
> **4. Lack of a Strong Baseline**
> The baseline used (SMA) is very weak. For a highly autocorrelated series, a Naive Baseline ($y_t = y_{t-1}$) achieves a test MAE of **0.1568 °C**. The Vanilla RNN only slightly improves on this naive baseline (0.1246 °C), and the LSTM is worse than the naive baseline.
> 
> **5. Out-of-Order Execution Bug**
> Cell 27 in the submission notebook contains a `NameError: name 'moving_average_forecast' is not defined` because it was run out of order without running Cell 16 first.

---

## 🔮 Future Improvements

1. **Fix Early Stopping**: Adjust early stopping patience and resolve validation dropout differences.
2. **Controlled Benchmark**: Align model structures (same depth, output heads, and dropout) to isolate recurrent cells.
3. **Incorporate Naive Baseline**: Compare models against a persistence baseline.
4. **Multivariate Inputs**: Add humidity, pressure, and wind velocity features to evaluate if LSTM benefits from richer input dynamics.

---

## 📂 Project Structure

```
.
├── Time_Series_Forecasting_P6_Submission.ipynb  # Main Jupyter Notebook
└── README.md                                    # Project documentation (this file)
```

---

## 🚀 Installation Guide

1. Clone or download this repository.
2. Install dependencies:
   ```bash
   pip install numpy pandas matplotlib scikit-learn torch kagglehub
   ```

---

## 💻 Usage Instructions

Open the notebook using Jupyter:
```bash
jupyter notebook Time_Series_Forecasting_P6_Submission.ipynb
```
Run the cells chronologically. Make sure to run Cell 16 before running Cell 27 to avoid the `NameError` execution order bug.

---

## 📝 Conclusion

The project successfully implements univariate temperature forecasting using PyTorch sequence models. Empirically, the simpler Vanilla RNN outperforms a regularized 2-layer LSTM on this 1-step prediction task, demonstrating that model complexity must align with task difficulty.
