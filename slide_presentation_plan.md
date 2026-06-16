# 🛝 Slide-by-Slide Presentation Plan

This plan details a structured, 14-slide presentation matching the exact contents and outcomes of the notebook.

---

### Slide 1: Title & Introduction
- **Title**: Benchmarking Sequence Models for Climate Time-Series Forecasting
- **Key Content**:
  - Task: 1-step-ahead univariate temperature prediction
  - Dataset: Jena Climate (420,551 observations, 10-minute intervals)
  - Models compared: Simple Moving Average, Vanilla RNN, LSTM
  - Presenter: [Candidate Name]
- **What to Say**: "Good morning. In this presentation, I will detail my empirical comparison of statistical and deep recurrent architectures for predicting temperature at a 10-minute resolution, evaluating the impact of historical context length on prediction accuracy."
- **Transition Line**: "Let's begin by discussing our research objectives and the criteria we used to measure success."

---

### Slide 2: Research Objectives & Success Criteria
- **Title**: Research Objectives & Success Criteria
- **Key Content**:
  - **Research Question**: How does historical context length (lookback window) affect forecasting accuracy?
  - **Goal**: Compare non-parametric baselines against recurrent networks.
  - **Metrics**: Optimize Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE) on test data.
  - **Interpretability**: Evaluate errors in original Celsius units.
- **What to Say**: "Our primary objective is to evaluate whether adding historical context improves or degrades recurrent model forecasts and to identify structural boundaries where lookback windows plateau."
- **Transition Line**: "To ground this research, let's examine the raw meteorological dataset."

---

### Slide 3: Jena Climate Dataset
- **Title**: The Jena Climate Dataset
- **Key Content**:
  - Max Planck Institute for Biogeochemistry records (2009–2016).
  - 420,551 records at 10-minute intervals.
  - Target: `T (degC)` (temperature) extracted as a univariate target.
  - Note: Erroneous sentinel values $-9999.0$ present in wind velocity variables (noted in EDA, unused in modeling).
- **What to Say**: "We utilize the Jena Climate Dataset, containing over 420,000 observations. To keep our focus on model structure, we extract temperature as a univariate series."
- **Transition Line**: "Let's explore the temporal patterns present in this temperature series."

---

### Slide 4: Exploratory Data Analysis (EDA)
- **Title**: Exploratory Data Analysis
- **Key Content**:
  - **Annual seasonality**: Temperature spans $-20\text{°C}$ to $+35\text{°C}$ across years.
  - **Diurnal cycle**: Daily oscillations clearly visible (July 2013 zoom-in).
  - **Related features**: Relative humidity and pressure plotted to confirm physical interactions.
- **What to Say**: "EDA reveals clear annual patterns and daily diurnal cycles, where temperature peaks in the afternoon and bottoms out before sunrise. This guides our lookback window selections."
- **Transition Line**: "Before feeding this data to neural networks, we must split and normalize it."

---

### Slide 5: Data Preprocessing
- **Title**: Preprocessing & Leakage Prevention
- **Key Content**:
  - **Chronological Split**: 70% Train (294,385), 15% Val (63,082), 15% Test (63,083).
  - No random shuffling to prevent temporal data leakage.
  - **MinMaxScaler Scaling**: Normalizes temperature values to $[0, 1]$.
  - Scaler fit *exclusively* on the training set to prevent range leakage.
- **What to Say**: "We split our data chronologically to preserve the flow of time and fit our scaler only on training data to avoid leaking future range limits."
- **Transition Line**: "Now, let's look at how we format this data into sequences for PyTorch."

---

### Slide 6: Sequence Generation
- **Title**: Sliding Window Data Loader
- **Key Content**:
  - Custom PyTorch `TimeSeriesDataset` implementation.
  - Input: $X = [t-L, \dots, t-1]$ (lookback window).
  - Target: $y = [t]$ (next step temperature).
  - Data shape: `(batch_size, lookback_steps, 1)`.
  - Mini-batches: Created with `batch_size = 64`.
- **What to Say**: "We transform our scaled arrays into sliding windows using a custom PyTorch dataset. This shapes our inputs into 3D tensors representing batches of historical windows."
- **Transition Line**: "With our pipeline set up, let's examine the three models we compared."

---

### Slide 7: Model 1: Simple Moving Average (Baseline)
- **Title**: Simple Moving Average Baseline
- **Key Content**:
  - Predicts the average of the lookback window: $\hat{y}_t = \frac{1}{L} \sum_{i=1}^L x_{t-i}$.
  - Zero parameters; requires no training.
  - Serves as the performance floor.
  - Disadvantage: Introduces a severe phase lag and dilutes recent relevant readings.
- **What to Say**: "Our first model is a zero-parameter moving average baseline, which serves as a basic sanity check."
- **Transition Line**: "Next, we transition to our first trainable sequence model: the Vanilla RNN."

---

### Slide 8: Model 2: Vanilla RNN Architecture
- **Title**: Vanilla RNN Architecture
- **Key Content**:
  - Single recurrent layer (`nn.RNN`) with 64 hidden units.
  - Tanh activation functions.
  - Linear output head: `Linear(64, 1)` mapped to the last hidden state $h_T$.
  - Total parameters: 4,353.
  - Pros/Cons: Lightweight and fast, but prone to vanishing gradients.
- **What to Say**: "The Vanilla RNN uses a single recurrent layer with 64 hidden units to pass temporal states forward. It has a tiny footprint of just over 4,000 parameters."
- **Transition Line**: "To address the memory limitations of the RNN, we also implement an LSTM."

---

### Slide 9: Model 3: LSTM Gated Architecture
- **Title**: LSTM Gated Architecture
- **Key Content**:
  - 2-layer stacked `nn.LSTM` with 64 hidden units.
  - Dropout regularization (0.2) between LSTM layers.
  - MLP output head: `Linear(64, 32)` -> `ReLU` -> `Linear(32, 1)`.
  - Total parameters: 51,489.
  - Pros/Cons: Gated memory prevents gradient decay, but it is far more complex and prone to overfitting.
- **What to Say**: "Our LSTM model is significantly deeper, utilizing two stacked layers, dropout, and a two-layer MLP head to capture complex long-term dependencies."
- **Transition Line**: "Let's review the training strategy and early stopping rules applied to these networks."

---

### Slide 10: Training Strategy
- **Title**: Model Training Configuration
- **Key Content**:
  - Loss function: Mean Squared Error (`nn.MSELoss`).
  - Optimizer: `Adam` with a learning rate of $1e-3$.
  - Early stopping: Validation loss monitored with a patience of 5 to 7 epochs.
  - Weights restored: Checkpoint with the lowest validation loss.
- **What to Say**: "We optimize our models using Adam and MSE loss, using early stopping to halt training once validation loss plateaus."
- **Transition Line**: "Now, let's review the metrics achieved in our default run."

---

### Slide 11: Benchmark Results (Lookback = 24 steps)
- **Title**: Performance Benchmark (4-Hour Lookback)
- **Key Content**:
  - SMA Baseline: MAE = 1.2014 °C, RMSE = 1.6382 °C
  - **Vanilla RNN**: MAE = **0.1250 °C**, RMSE = **0.1954 °C**
  - LSTM Model: MAE = 0.2981 °C, RMSE = 0.4257 °C
  - *Note*: Vanilla RNN outperforms the LSTM by a wide margin.
- **What to Say**: "At a 4-hour lookback, both neural models beat the SMA baseline. However, the Vanilla RNN significantly outperforms the LSTM, achieving an MAE of 0.1250 °C."
- **Transition Line**: "Let's see if this relationship holds across our lookback window sweep."

---

### Slide 12: Lookback Sweep Results
- **Title**: Lookback Window Sweep Metrics
- **Key Content**:
  - Lookback windows: 24 (4h), 72 (12h), 144 (24h), and 288 (48h) steps.
  - RNN MAE remains stable: ~0.1246 °C to 0.1368 °C.
  - LSTM MAE remains higher: ~0.2306 °C to 0.5213 °C.
  - Moving Average degrades rapidly as window length increases.
- **What to Say**: "Across all lookback lengths, the Vanilla RNN continues to lead. The optimal lookback is 144 steps (24 hours), matching the daily diurnal cycle."
- **Transition Line**: "These results expose some critical bugs and contradictions in the notebook."

---

### Slide 13: Critical Callouts & Pipeline Anomaly
- **Title**: Pipeline Anomaly & Code Contradictions
- **Key Content**:
  - **Contradiction**: Notebook text claims LSTM outperforms RNN, but data proves the opposite.
  - **Early Stopping Bug**: LSTM early stopped at Epoch 8 and reverted to Epoch 1, leaving the model under-trained.
  - **Unfair Comparison**: LSTM model has 2 layers, dropout, and an MLP head, while RNN is simple.
  - **Baseline Choice**: A simple Naive Baseline ($y_t = y_{t-1}$) yields an MAE of **0.1568 °C**.
- **What to Say**: "Our analysis revealed that the LSTM's poor performance was due to early stopping restoring an under-trained Epoch 1 state. Additionally, when compared to a strong Naive baseline, the neural networks' value-add is much smaller than claimed."
- **Transition Line**: "Let's summarize how we plan to address these findings."

---

### Slide 14: Conclusion & Future Work
- **Title**: Conclusion & Future Work
- **Key Content**:
  - **Takeaway**: Simple architectures (RNN) perform exceptionally well on 1-step forecasting.
  - **Fixes**: Adjust early stopping patience and equalize architectural comparisons.
  - **Expansion**: Benchmark against Naive baseline and test on multivariate data.
- **What to Say**: "In conclusion, simple recurrent models excel at 1-step temperature prediction, but hyperparameter configurations and baseline choices require careful auditing to avoid false conclusions."
- **Transition Line**: "Thank you. I am happy to open the floor to any questions."

---

## ⚠️ Challenges & Limitations (CRITICAL CALLOUTS)

> [!WARNING]
> **1. Performance Contradiction**: The slides address the clear gap between the notebook's claims of LSTM superiority and the actual numbers showing RNN dominates.
> **2. Baseline Flaw**: SMA baseline averages out recent details, which is highly ineffective for autocorrelated climate trends. Naive baseline is the proper baseline.
