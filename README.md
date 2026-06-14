# 📈 Climate Time-Series Forecasting: RNN vs. LSTM vs. Moving Average

An in-depth empirical comparison of statistical baselines and deep recurrent architectures (Vanilla RNN and LSTM) for univariate, 1-step-ahead temperature forecasting on the **Jena Climate Dataset**.

---

## 📋 Table of Contents
1. [Project Overview](#-project-overview)
2. [Quick Results & Insights](#-quick-results--insights)
3. [The "Whys": Core Motivation & Theory](#-the-whys-core-motivation--theory)
   - [Why Temperature Forecasting?](#why-temperature-forecasting)
   - [Why the Moving Average Baseline?](#why-the-moving-average-baseline)
   - [Why the Vanishing Gradient Problem in RNNs?](#why-the-vanishing-gradient-problem-in-rnns)
   - [Why LSTMs Solve the Memory Problem](#why-lstms-solve-the-memory-problem)
   - [Why Lookback Window Length Matters](#why-lookback-window-length-matters)
   - [Why did Vanilla RNN slightly outperform LSTM in our sweep?](#why-did-vanilla-rnn-slightly-outperform-lstm-in-our-sweep)
4. [The "Hows": Data Engineering & Preprocessing](#-the-hows-data-engineering--preprocessing)
   - [Dataset Collection & Subsampling](#dataset-collection--subsampling)
   - [Chronological Split & Scaling](#chronological-split--scaling)
   - [Sliding Window Processing](#sliding-window-processing)
5. [Model Architectures & Training](#-model-architectures--training)
   - [Model Architectures](#model-architectures)
   - [Training Strategy](#training-strategy)
6. [Interactive Code Links](#-interactive-code-links)
7. [Visualizing the Results](#-visualizing-the-results)
8. [Reproducing the Results](#-reproducing-the-results)

---

## 🔍 Project Overview

The goal of this project is to predict the temperature ($T$) of the next hour based on historical hourly readings. We use the **Jena Climate Dataset** (Max Planck Institute for Biogeochemistry), containing 7 years of 10-minute meteorological measurements. 

We downsample the data to **hourly** intervals and compare three distinct approaches:
1. **Moving Average Baseline**: A zero-parameter statistical benchmark.
2. **Vanilla RNN**: A recurrent network using simple tanh transitions.
3. **LSTM (Long Short-Term Memory)**: A gated recurrent network designed to capture long-term context.

We also conduct a **hyperparameter sweep** over the lookback window length ($6\text{h}$, $12\text{h}$, $24\text{h}$, $48\text{h}$, $72\text{h}$, and $120\text{h}$) to evaluate how much historical context neural networks need to minimize prediction error.

---

## ⚡ Quick Results & Insights

### Final Model Comparison (Lookback = 24h)
Evaluated on the test dataset (representing the final 15% of the timeline):

| Model | MAE (°C) | RMSE (°C) | Summary / Performance |
| :--- | :---: | :---: | :--- |
| **Moving Average** | 2.4920 | 3.1680 | Baseline threshold (poor tracking of dynamic shifts) |
| **Vanilla RNN** | **0.5043** | **0.7220** | Best performance (highly responsive, low complexity) |
| **LSTM** | 0.5165 | 0.7400 | Highly competitive (slightly heavier parameter footprint) |

### Lookback Sweep Summary (MAE in °C)
| Lookback (hours) | Moving Average | Vanilla RNN | LSTM |
| :---: | :---: | :---: | :---: |
| **6h** | 1.8477 | 0.5276 | 0.6046 |
| **12h** | 2.6531 | 0.5073 | 0.5516 |
| **24h** | 2.4920 | 0.5013 | 0.5361 |
| **48h** | 2.8236 | **0.4925** | 0.5502 |
| **72h** | 3.0447 | 0.5036 | 0.6087 |
| **120h** | 3.3434 | 0.4997 | 0.5897 |

> [!TIP]
> The sweet spot for the lookback window is **24 to 48 hours**. Beyond 48 hours, model performance plateaus or degrades, suggesting that context older than two days adds noise rather than signal for next-hour predictions.

---

## 🧠 The "Whys": Core Motivation & Theory

### Why Temperature Forecasting?
Temperature forecasting is critical for energy grid load balancing, agricultural scheduling, smart HVAC systems, and climate change monitoring. From a modeling perspective, temperature series present a combination of:
* **Short-term persistence** (it is likely to be similar to an hour ago).
* **Diurnal cycles** (warm days, cool nights).
* **Seasonal trends** (hot summers, cold winters).

### Why the Moving Average Baseline?
Without a baseline, a neural network’s error metrics are meaningless. In time series, the simplest forecast is persistence ($x_{t+1} = x_t$) or moving average. Showing that a complex neural network dramatically beats a moving average (reducing MAE from $\approx 2.49\text{°C}$ to $\approx 0.50\text{°C}$) proves the network is actually learning meteorological dynamics rather than simple averages.

### Why the Vanishing Gradient Problem in RNNs?
Standard Recurrent Neural Networks process sequences by repeating the recurrence relation:
$$h_t = \tanh(W_{hh} h_{t-1} + W_{xh} x_t + b_h)$$
When backpropagating through time ($T$ steps), the gradient of the loss with respect to the initial hidden state includes a product of Jacobian matrices:
$$\frac{\partial Loss}{\partial h_0} \propto \prod_{k=1}^T \frac{\partial h_k}{\partial h_{k-1}}$$
If the eigenvalues of $W_{hh}$ are less than 1, the gradient decays exponentially as $T$ increases. As a result, the model forgets distant history and is updated almost entirely by recent observations.

### Why LSTMs Solve the Memory Problem
LSTMs introduce a **Cell State ($C_t$)** that runs straight down the chain with only linear interactions, acting as a high-speed highway for memory. This highway is regulated by three neural gates:
1. **Forget Gate ($f_t$)**: Decides what percentage of the past cell state to keep:
   $$f_t = \sigma(W_f [h_{t-1}, x_t] + b_f)$$
2. **Input Gate ($i_t$)**: Selects which new inputs to write into the cell state:
   $$i_t = \sigma(W_i [h_{t-1}, x_t] + b_i)$$
   $$\tilde{C}_t = \tanh(W_c [h_{t-1}, x_t] + b_c)$$
   $$C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$$
3. **Output Gate ($o_t$)**: Filters the cell state to produce the next hidden state $h_t$:
   $$o_t = \sigma(W_o [h_{t-1}, x_t] + b_o)$$
   $$h_t = o_t \odot \tanh(C_t)$$

This design allows gradients to flow back through the cell state linearly, preventing the exponential decay of gradients.

```
          [ LSTM Cell Architecture Flow ]
 
            Cell State (C_t-1) ──────────[ x ]──────────(+)──────────► Cell State (C_t)
                                          ▲             ▲
                                       Forget         Input
                                        Gate          Gate
                                          ▲             ▲
  Hidden State (h_t-1) ──┬───────► [ Gates & Activation ] ───────────► Hidden State (h_t)
                         │                ▲
  Current Input (x_t) ───┘                │
                                      Output Gate
```

### Why Lookback Window Length Matters
The lookback window defines the length of historical context fed to the model.
* **Too short (<12h)**: The model cannot observe the full diurnal (day/night) cycle, making it blind to whether the temperature is rising due to morning sunrise or falling in the evening.
* **Too long (>48h)**: The model receives redundant or noisy historical states. The optimization space becomes unnecessarily large, increasing training times and risk of overfitting.

### Why did Vanilla RNN slightly outperform LSTM in our sweep?
Intuitively, we expect LSTMs to perform better than Vanilla RNNs on long sequences. However, in our empirical tests:
1. **Task Simplicity (1-step prediction)**: Univariate 1-step forecasting is heavily dominated by immediate auto-regressive values. The model mostly needs to know what happened in the last 1 to 3 hours.
2. **Vanishing Gradient as a Regularizer**: The Vanilla RNN’s natural decay of older states acts as an inductive bias, focusing the model's updates on recent history.
3. **Overfitting & Regularization in LSTM**: The LSTM model is deeper (2 layers vs. 1 layer), contains significantly more parameters ($\approx 50\text{k}$ vs. $\approx 4\text{k}$), and includes a $0.2$ dropout rate. On a simple 1D time series, the LSTM is prone to slight underfitting or overparameterization, requiring more meticulous hyperparameter tuning (e.g. learning rates, batch sizes, or epochs) to beat the simpler RNN.

---

## 🛠️ The "Hows": Data Engineering & Preprocessing

### Dataset Collection & Subsampling
The original dataset records measurements at 10-minute intervals. 
* **Subsampling**: We downsample the data by selecting every 6th row (`iloc[::6]`) to create an hourly dataset. This drastically reduces high-frequency noise, cuts the training dataset size from $420,000$ to $70,087$ samples, and accelerates training loops by a factor of 6.

### Chronological Split & Scaling
* **Splitting**: We perform a chronological split ($70\%$ Train, $15\%$ Validation, $15\%$ Test). We avoid random shuffling during splitting because shuffling would leak future climate patterns into past evaluations.
* **Scaling**: We fit a `MinMaxScaler(feature_range=(0,1))` on the **training set only**. Using the validation or test sets to determine the scale parameters ($min, max$) would introduce **data leakage** (spilling future range information into the historical training context).

### Sliding Window Processing
The dataset is converted into training frames using a sliding window:
```
Given a sequence: [10.2, 10.5, 11.0, 11.2, 11.8, 12.1] with lookback = 4
  - Input X_1: [10.2, 10.5, 11.0, 11.2]  →  Target y_1: [11.8]
  - Input X_2: [10.5, 11.0, 11.2, 11.8]  →  Target y_2: [12.1]
```
This transformation is implemented efficiently inside the PyTorch `TimeSeriesDataset` class:
* Input dimensions: `(N_samples, lookback_window, 1)`
* Target dimensions: `(N_samples, 1)`

---

## ⚙️ Model Architectures & Training

### Model Architectures

#### 1. Vanilla RNN
* **Recurrent Layer**: A single-layer PyTorch `nn.RNN` with `hidden_size=64` and `tanh` activation.
* **Head**: A fully connected layer projecting the final hidden state $h_T$ to a single target: `nn.Linear(64, 1)`.
* **Total Parameters**: $4,353$

#### 2. LSTM Model
* **Recurrent Layer**: A 2-layer stacked `nn.LSTM` with `hidden_size=64` and `dropout=0.2`.
* **Head**: Fully connected hidden layer mapping `64 -> 32`, followed by a `ReLU` activation, and a final projection layer mapping `32 -> 1`.
* **Total Parameters**: $51,489$

### Training Strategy
* **Loss Function**: Mean Squared Error (`nn.MSELoss()`). It penalizes larger deviations quadratically, prompting models to avoid extreme forecasting errors.
* **Optimization**: `Adam` optimizer with a learning rate of `1e-3` and **Gradient Clipping** (`max_norm=1.0`) to suppress exploding gradients during RNN updates.
* **Learning Rate Scheduling**: `ReduceLROnPlateau` monitors the validation loss. If it plateaus for 3 epochs, the learning rate is halved to explore finer local minima.
* **Early Stopping**: The training loop tracks validation loss and terminates if no improvement is observed for 7 consecutive epochs (saving the best-performing weights).

---

## 🔗 Interactive Code Links

The modeling pipelines, visual sweeps, and evaluations are fully documented in the following project files:

* 📊 **Jupyter Notebook**: [time_series_forecasting.ipynb](file:///c:/Users/Mohit-PC/Downloads/NN-12/time_series_forecasting.ipynb) — Contains the complete implementation from EDA to final model evaluation.
* 📈 **Final Metrics**: [final_metrics.csv](file:///c:/Users/Mohit-PC/Downloads/NN-12/final_metrics.csv) — Stores the test MAE & RMSE values for lookback=24h.
* 🗃️ **Lookback Sweeps**: [lookback_analysis.csv](file:///c:/Users/Mohit-PC/Downloads/NN-12/lookback_analysis.csv) — Stores performance metrics across different lookback windows.
* 💾 **Saved RNN Weights**: [rnn_model.pth](file:///c:/Users/Mohit-PC/Downloads/NN-12/rnn_model.pth)
* 💾 **Saved LSTM Weights**: [lstm_model.pth](file:///c:/Users/Mohit-PC/Downloads/NN-12/lstm_model.pth)

---

## 🎨 Visualizing the Results

The training and evaluation scripts output several analytical plots. You can review them directly:

### 1. Exploratory Data Analysis
[01_eda.png](file:///c:/Users/Mohit-PC/Downloads/NN-12/01_eda.png)
* *Top*: The full 7-year raw temperature trend showing seasonal fluctuations.
* *Middle*: A zoom-in of July 2013 highlighting the diurnal cycle (temperature peaks in the afternoon, troughs before sunrise).
* *Bottom*: A clean histogram showing the normal distribution of temperature centered around $9.4\text{°C}$.

### 2. Training Convergence
[02_training_curves.png](file:///c:/Users/Mohit-PC/Downloads/NN-12/02_training_curves.png)
* Depicts the training vs. validation MSE losses across epochs. It validates that early stopping successfully halted training before the networks overfit.

### 3. Model Error Comparison
[03_model_comparison.png](file:///c:/Users/Mohit-PC/Downloads/NN-12/03_model_comparison.png)
* A bar chart illustrating MAE and RMSE errors on the test set for the default 24h lookback window.

### 4. Forecast Visualizations (14-Day Outlook)
[04_forecast_comparison.png](file:///c:/Users/Mohit-PC/Downloads/NN-12/04_forecast_comparison.png)
* Superimposes prediction curves over actual ground truth data for a 2-week period. The neural network predictions trace the true curve closely, while the Moving Average displays a visible phase delay.

### 5. Prediction Residuals
[05_residuals.png](file:///c:/Users/Mohit-PC/Downloads/NN-12/05_residuals.png)
* Histograms of residuals ($Actual - Predicted$). The neural networks exhibit tight, narrow bell curves centered at $0.0\text{°C}$, demonstrating unbiased predictions.

### 6. Lookback Windows Comparison
[06_lookback_analysis.png](file:///c:/Users/Mohit-PC/Downloads/NN-12/06_lookback_analysis.png)
* Line plots illustrating how MAE and RMSE evolve as a function of the lookback window.

### 7. Performance Heatmap
[07_mae_heatmap.png](file:///c:/Users/Mohit-PC/Downloads/NN-12/07_mae_heatmap.png)
* A 2D grid mapping models against lookback windows, color-coded by MAE.

### 8. Architectural Concepts
[08_architecture_diagram.png](file:///c:/Users/Mohit-PC/Downloads/NN-12/08_architecture_diagram.png)
* A diagram comparing simple RNN linear recurrent transitions against the gated memory architecture of the LSTM.

### 9. Prediction Scatter Plots
[09_scatter_plots.png](file:///c:/Users/Mohit-PC/Downloads/NN-12/09_scatter_plots.png)
* Plotting $Predicted$ vs $Actual$ values. Perfect forecasts follow the $x=y$ identity diagonal line. The RNN and LSTM models tightly align with the diagonal, whereas the Moving Average displays significant dispersion.

---

## 🚀 Reproducing the Results

To set up the environment and execute the pipeline:

### 1. Requirements
Ensure you have Python 3.8+ installed along with the following packages:
```bash
pip install torch numpy pandas matplotlib scikit-learn jupyter
```

### 2. Dataset Setup
Download the **Jena Climate Dataset** from Kaggle:
* Kaggle Link: [jena-climate](https://www.kaggle.com/datasets/mnassrib/jena-climate)
* Unzip and place `jena_climate_2009_2016.csv` in the root of the project directory.

### 3. Execution
Launch Jupyter Notebook and run the cells sequentially:
```bash
jupyter notebook time_series_forecasting.ipynb
```
Running the notebook will reproduce the datasets, train the models, output the `.pth` weights, and generate all visualizations from scratch.
