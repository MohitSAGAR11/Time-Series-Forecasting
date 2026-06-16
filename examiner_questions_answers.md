# ❓ Examiner Questions and Answers

This document lists 40+ detailed questions and answers categorized to assist in preparing for an academic viva or project defense.

---

## Category A: Basic Concepts
1. **Question: What is the main objective of this machine learning project?**
   * *Answer:* The objective is to forecast the temperature of the next 10-minute interval (1-step-ahead prediction) using historical temperature data from the Jena Climate Dataset. The project evaluates how lookback window length affects forecast accuracy across Simple Moving Average, Vanilla RNN, and LSTM models.
2. **Question: What is a univariate time series?**
   * *Answer:* A univariate time series consists of observations of a single variable recorded sequentially over time. In this project, the only variable used as both input and target is temperature (`T (degC)`).
3. **Question: What dataset was used in this study, and what is its time interval?**
   * *Answer:* The Jena Climate Dataset recorded by the Max Planck Institute for Biogeochemistry was used. The measurements are recorded at 10-minute intervals from 2009 through 2016.
4. **Question: What is a "lookback window"?**
   * *Answer:* A lookback window represents the length of historical steps fed to the model to make a prediction. In this project, lookback windows of 24, 72, 144, and 288 steps correspond to 4 hours, 12 hours, 24 hours, and 48 hours of climate history, respectively.
5. **Question: What is the difference between training, validation, and test datasets?**
   * *Answer:* The training set (70%) is used to update model weights. The validation set (15%) is used to monitor overfitting and trigger early stopping. The test set (15%) is used for final evaluation.
6. **Question: What is the purpose of a baseline model in time-series forecasting?**
   * *Answer:* A baseline model serves as a simple benchmark. If a complex neural network cannot outperform a zero-parameter baseline, it indicates that the neural network has not learned meaningful patterns.

---

## Category B: Data Preprocessing & Pipeline
7. **Question: Why was the data split chronologically rather than randomly?**
   * *Answer:* In time-series forecasting, random shuffling violates the temporal ordering of the data and introduces future-to-past data leakage. Chronological splitting ensures that the model is always evaluated on data that comes after the training history, mimicking real-world deployment.
8. **Question: Why is feature scaling applied, and which scaler did you use?**
   * *Answer:* Scaling standardizes input ranges, which prevents large values from destabilizing gradient updates in neural networks. This project uses scikit-learn's `MinMaxScaler` to scale temperature values to a $[0, 1]$ range.
9. **Question: Why was the scaler fit only on the training set?**
   * *Answer:* Fitting on the training set ensures the scale parameters ($min$ and $max$) are calculated without knowing validation or test values. Fitting on all data would leak future range information.
10. **Question: How does the PyTorch `TimeSeriesDataset` structure inputs?**
    * *Answer:* It uses a sliding window loop. For each index $i$ starting at the lookback length, it slices history from $i-lookback$ to $i$ as input $X$, and sets index $i$ as target $y$. It returns these as float32 tensors.
11. **Question: What is the dimension of the batches returned by the DataLoader?**
    * *Answer:* The batch input tensor shape is `(64, lookback, 1)` and the target tensor shape is `(64, 1)`, where 64 is the batch size, `lookback` is the window size, and 1 represents our single temperature feature.
12. **Question: What is the impact of sequence generation on validation data size?**
    * *Answer:* The first `lookback` steps of the validation and test datasets are not predicted because they lack sufficient historical context within their sliced arrays. This slightly reduces the actual number of evaluated sequences by $lookback$ steps.

---

## Category C: Technical & Deep Learning
13. **Question: What loss function was used to train the networks, and why?**
    * *Answer:* Mean Squared Error (`nn.MSELoss()`) was used. It calculates the average squared difference between predictions and targets, penalizing larger errors more heavily.
14. **Question: What optimizer did you use, and what was the learning rate?**
    * *Answer:* The Adam optimizer was used with a learning rate of $1e-3$. Adam combines momentum and adaptive learning rates for efficient training.
15. **Question: How does early stopping work in this pipeline?**
    * *Answer:* The training loop tracks validation MSE. If validation loss does not improve for a set number of epochs (patience = 7 in the single run, 3 in the sweep), training halts and the best weights are restored.
16. **Question: Why are predictions inverse-transformed back to Celsius?**
    * *Answer:* Normalization compresses values to $[0, 1]$. To calculate meaningful real-world errors (MAE and RMSE in degrees Celsius), predictions must be scaled back using the training scaler's parameters.
17. **Question: Why does the validation loss in Epoch 1 start lower than training loss?**
    * *Answer:* In training mode, dropout (0.2) is active in the LSTM, which adds noise and increases training loss. In validation mode, dropout is deactivated, which reduces prediction noise and lowers validation loss.
18. **Question: What device was used to train the models in this execution?**
    * *A:* The PyTorch device selection fell back to `'cpu'` because CUDA was not available on the execution host, as shown in the environment setup outputs.

---

## Category D: Model-Specific Details
19. **Question: What is the exact layer structure of the Vanilla RNN?**
    * *Answer:* The Vanilla RNN consists of a single `nn.RNN` layer with `input_size=1`, `hidden_size=64`, and `num_layers=1` with tanh activation. The final hidden state $h_T$ is passed to a linear layer `nn.Linear(64, 1)`.
20. **Question: What is the exact layer structure of the LSTM model?**
    * *Answer:* The LSTM model consists of a 2-layer stacked `nn.LSTM` with `hidden_size=64` and `dropout=0.2`. The final hidden state is passed through `nn.Dropout(0.2)`, a linear layer `nn.Linear(64, 32)`, a `nn.ReLU()` activation, and a final linear layer `nn.Linear(32, 1)`.
21. **Question: Why does the LSTM model contain so many more parameters than the RNN?**
    * *Answer:* The Vanilla RNN has 4,353 parameters. The LSTM has 51,489 parameters because LSTMs use four internal gating equations (forget, input, cell, output gates) per layer, stacked across 2 layers, alongside a two-layer output head.
22. **Question: Why did the LSTM model perform poorly compared to the Vanilla RNN?**
    * *Answer:* In the notebook, early stopping triggered at Epoch 8 for the LSTM because validation loss rose after Epoch 1. Reverting the model to Epoch 1 weights left it under-trained, leading to poor test metrics.
23. **Question: How does the vanishing gradient problem affect Vanilla RNNs?**
    * *Answer:* Standard RNNs backpropagate gradients through time by multiplying weight jacobians. Over long sequences, if the weights are small, these gradients decay exponentially, causing the network to forget past information.
24. **Question: How does the LSTM architecture mitigate vanishing gradients?**
    * *Answer:* LSTMs introduce a cell state ($C_t$) that runs linearly through the sequence with minimal additive modifications. This creates a direct path for gradients to flow backward without exponential decay.
25. **Question: What role does dropout play in the LSTM model?**
    * *Answer:* Dropout randomly sets 20% of the activations to zero during training. This regularizes the network, preventing it from co-adapting features and reducing overfitting.

---

## Category E: Time-Series-Specific Questions
26. **Question: What is autocorrelation, and why is it important here?**
    * *Answer:* Autocorrelation measures the correlation of a signal with a delayed copy of itself. Temperature has very high short-term autocorrelation (e.g. 10 minutes apart), making recent values strong predictors of the next step.
27. **Question: Why does the Simple Moving Average baseline perform worse as lookback increases?**
    * *Answer:* A moving average predicts the mean of the window. As the window expands, it includes older values (e.g. from 12 or 24 hours ago) that are very different from the current temperature, increasing prediction lag.
28. **Question: Why is 144 steps (24 hours) a critical lookback window size?**
    * *Answer:* 144 steps correspond to a full diurnal (24-hour) cycle. This window size provides the model with a complete day's history, allowing it to capture temperature rises and falls.
29. **Question: How did the lookback sweep show Vanilla RNN handling different history lengths?**
    * *Answer:* The Vanilla RNN's MAE remained stable around 0.125 °C to 0.136 °C across all windows (24 to 288 steps), showing that it successfully extracted relevant short-term features regardless of window size.
30. **Question: What is the "naive baseline" for this dataset, and what is its MAE?**
    * *Answer:* The naive baseline predicts the most recent temperature value ($y_t = y_{t-1}$). On this test set, it achieves a strong MAE of **0.1568 °C** due to high autocorrelation.
31. **Question: Why does the SMA baseline achieve a lower error at lookback 144 than at lookback 72?**
    * *Answer:* At 144 steps (24 hours), the window covers a full diurnal cycle. Because temperature fluctuates daily, the average of a full 24-hour cycle is closer to the current value than the average of a 12-hour cycle (which is biased by only daytime or nighttime readings).

---

## Category F: Forecasting & Evaluation
32. **Question: What was the lowest MAE achieved in this notebook, and by which model?**
    * *Answer:* The lowest MAE was **0.1246 °C**, achieved by the Vanilla RNN at a lookback window of 144 steps (24 hours) in the sweep loop.
33. **Question: What was the LSTM's lowest MAE in the lookback sweep?**
    * *Answer:* The LSTM's lowest MAE in the lookback sweep was **0.2306 °C** at a lookback of 72 steps (12 hours). Its error rose at longer lookback windows.
34. **Question: What is the difference between MAE and RMSE?**
    * *Answer:* MAE measures the average absolute difference, giving equal weight to all errors. RMSE squares errors before averaging, penalizing larger deviations more heavily.
35. **Question: Why is the LSTM's test MAE of 0.2981 °C in the standalone run better than its MAE of 0.5213 °C in the sweep?**
    * *Answer:* In the standalone run, the model trained for up to 50 epochs with a patience of 7. In the sweep, it was restricted to 10 epochs with a patience of 3. The sweep models were under-trained.
36. **Question: What is the main contradiction in the notebook's final summary text?**
    * *Answer:* The text claims that the "LSTM consistently outperforms both baselines." However, the metrics show the Vanilla RNN outperforming the LSTM in all tests (e.g. RNN MAE 0.1250 °C vs. LSTM MAE 0.2981 °C).

---

## Category G: Deployment & Practical Engineering
37. **Question: How would you resolve the out-of-order execution NameError in Cell 27?**
    * *Answer:* Ensure that all cell definitions (specifically Cell 16, which defines `moving_average_forecast`) are executed in order before running the lookback sweep.
38. **Question: If deployed to production, what is the inference frequency of this pipeline?**
    * *Answer:* Because the dataset records measurements at 10-minute intervals, the model must run inference every 10 minutes to predict the next step.
39. **Question: Would you recommend the LSTM model for production based on these results?**
    * *Answer:* No. The Vanilla RNN is faster, has 90% fewer parameters, and achieves a lower MAE (0.1250 °C vs. 0.2981 °C). The LSTM is overparameterized and unstable under the current configuration.
40. **Question: What is the risk of training on CPU instead of GPU for this project?**
    * *Answer:* The only risk is training speed. The dataset is moderately large, so training multiple recurrent models across sweeps on a CPU is slow, though it does not affect model accuracy.

---

## ⚠️ Challenges & Limitations (CRITICAL CALLOUTS)

> [!WARNING]
> **1. Contradiction Flag**: Examiners will notice that the model summary says LSTM outperforms baseline/RNN, while the actual data printed shows RNN performs significantly better.
> **2. Early Stopping Defect**: The LSTM early stopping triggered too early because validation loss rose after epoch 1, leading to evaluating a model that was not converged.
> **3. Weak Baseline**: SMA is a weak baseline for highly autocorrelated time series. A naive persistence baseline of 0.1568 °C should be added.
