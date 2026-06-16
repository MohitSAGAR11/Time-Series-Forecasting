# 🎙️ Viva Oral Examination Presentation Script

This document provides a first-person, conversational script for a 15-minute viva oral presentation based on the sequence modeling benchmarks.

---

## Part 1: Introduction
"Good morning, esteemed committee members. Today, I am presenting my time-series forecasting project, which focuses on benchmarks for univariate 1-step-ahead temperature prediction. The central research question of this project is: *How does the length of historical context, or lookback window, affect forecast accuracy, and does it expose structural limitations in recurrent models?* Using the Jena Climate Dataset, I compare three different forecasting models—a Simple Moving Average statistical baseline, a Vanilla RNN, and a Long Short-Term Memory network—across history lengths ranging from 4 hours to 48 hours."

### Sample Examiner Q&A
* **Question 1: Why did you limit this study to univariate forecasting rather than multivariate?**
  * *Answer:* The primary objective was to isolate and evaluate the structural differences between recurrent neural architectures (Vanilla RNN vs. LSTM) and lookback windows. Introducing multivariate features like pressure and humidity would introduce complex feature interactions, making it harder to determine if performance changes were due to architecture or feature quality.
* **Question 2: How is a '1-step-ahead' task defined in this dataset?**
  * *Answer:* The dataset is recorded at 10-minute intervals. Therefore, a 1-step-ahead forecast means the model receives a historical sequence of temperatures and predicts the temperature for the very next 10-minute step.

---

## Part 2: Dataset Description
"Let's look at our dataset. I utilized the Jena Climate Dataset from the Max Planck Institute. It contains 420,551 measurements recorded at 10-minute intervals from 2009 through 2016. The target is the temperature column, `T (degC)`, which I extracted as a univariate target. The date range spans from January 1, 2009, to January 1, 2017, representing 7 full years of record."

### Sample Examiner Q&A
* **Question 1: What was the physical resolution of the data, and did you subsample it?**
  * *Answer:* The physical resolution is 10-minute intervals. Unlike older iterations of this project which subsampled the data to hourly intervals, this notebook models the full 420,551 observations at their original 10-minute resolution to retain the highest-frequency dynamics.
* **Question 2: Were there any missing values or anomalies in the dataset?**
  * *Answer:* There were zero missing values in the target column. However, during exploratory analysis, I noted that wind velocity columns (`wv (m/s)` and `max. wv (m/s)`) contained sentinel values of `-9999.0` for sensor errors. Since this project is strictly univariate, these anomalies did not impact our temperature pipeline.

---

## Part 3: Exploratory Data Analysis (EDA)
"In the exploratory phase, I constructed two specific visualizations for the temperature. The first is a full 7-year timeline plot of `T (degC)`. This plot displays a massive annual seasonality—with peaks in summer and troughs in winter—spanning approximately $-20\text{°C}$ to $+35\text{°C}$. The second plot zooms in on a single month, July 2013. In this zoom-in, a very strong diurnal, or daily, cycle is clearly visible. The temperature peaks in the mid-afternoon and hits its lowest point just before sunrise. I also plotted relative humidity and atmospheric pressure to confirm their seasonal cycles, though I proceeded with temperature alone."

### Sample Examiner Q&A
* **Question 1: Why did you plot a zoomed-in view of July 2013?**
  * *Answer:* Zooming into a single month reveals the diurnal cycle, showing that temperature fluctuates in a regular daily pattern. This is critical for choosing lookback windows, as it suggests that context lengths corresponding to multiples of 24 hours (144 steps) carry a strong physical signal.
* **Question 2: What did the relative humidity and pressure plots show?**
  * *Answer:* They confirmed that humidity oscillates inversely with temperature—peaking at night when it's cold and dropping during the day—while atmospheric pressure exhibits less high-frequency daily oscillation but has larger multi-day shifts corresponding to weather fronts.

---

## Part 4: Data Preprocessing
"To evaluate our models rigorously, I divided the 420,551 rows chronologically: 70% for training, 15% for validation, and 15% for testing. I fit a `MinMaxScaler` exclusively on the training set to scale values between 0 and 1, and then transformed the validation and test sets. To feed the sequence models, I implemented a sliding-window `TimeSeriesDataset` in PyTorch, which groups the data into input tensors of shape `(batch, lookback, 1)` and targets of shape `(batch, 1)`."

### Sample Examiner Q&A
* **Question 1: Why did you split the data chronologically rather than using a random split?**
  * *Answer:* In time-series forecasting, random shuffling violates the temporal ordering of the data and introduces future-to-past data leakage. Chronological splitting ensures that the model is always evaluated on data that comes after the training history, mimicking real-world deployment.
* **Question 2: Why must the MinMaxScaler be fit only on the training set?**
  * *Answer:* If we fit the scaler on the entire dataset, we leak information about the global minimum and maximum values of the validation and test periods into our training scale parameters. This is a subtle form of data leakage.

---

## Part 5: Model Architectures
"I evaluated three different modeling strategies, each representing a step up in complexity. 
First, the **Simple Moving Average baseline** predicts the average temperature of the lookback window. Think of this like driving a car forward by only looking at the average curvature of the road you traveled over the last few miles. It reacts slowly and lags behind.
Second, the **Vanilla RNN** processes the sequence step-by-step. Think of it like someone writing down the current temperature on a sticky note, mixing it with their memory of the last step, and writing a new note. It updates quickly but scribbles over old notes, forgetting distant history.
Third, the **LSTM** uses a dedicated memory line—the cell state—regulated by forget, input, and output gates. Think of the LSTM like a structured diary. Its gates act as editors, deciding what to remember, what to discard, and what to write down for the final prediction."

### Sample Examiner Q&A
* **Question 1: How did the layer configurations differ between your Vanilla RNN and LSTM models?**
  * *Answer:* The Vanilla RNN used a single recurrent layer with 64 hidden units and a direct linear projection head. The LSTM used 2 stacked layers with 64 hidden units, 0.2 dropout, and a deeper MLP output head consisting of a linear layer of 32 units, a ReLU activation, and a final linear layer.
* **Question 2: What are the total parameters of these two networks?**
  * *Answer:* The Vanilla RNN is highly compact with 4,353 parameters. The LSTM is significantly larger, containing 51,489 parameters due to its gated cells, stacked layers, and deeper output head.

---

## Part 6: Model Training & Evaluation
"For training, I used Mean Squared Error as the loss function and the Adam optimizer with a learning rate of $1e-3$. To prevent overfitting, I used early stopping on validation loss. During training, early stopping tracks the best validation loss and terminates training if no improvement is seen for a set number of epochs. Predictions and targets are inverse-transformed back to original Celsius units before computing the final Mean Absolute Error and Root Mean Squared Error."

### Sample Examiner Q&A
* **Question 1: Why is it important to evaluate performance on inverse-transformed data?**
  * *Answer:* Evaluating on scaled data between 0 and 1 is mathematically convenient for training, but it makes errors hard to interpret. Inverse-transforming the predictions back to Celsius allows us to report errors in real-world degrees, which is direct and meaningful for meteorologists.
* **Question 2: How did the validation loss behave during the training of the LSTM model?**
  * *Answer:* In Cell 21, the LSTM validation loss started at 0.000150 in Epoch 1, but rose to 0.000427 in Epoch 5. Because validation loss never dropped below the Epoch 1 value, early stopping triggered at Epoch 8 and restored the weights of Epoch 1.

---

## Part 7: Forecast Results & Sweep Analysis
"In our default run with a lookback of 24 steps, the Simple Moving Average achieved an MAE of 1.2014 °C. The Vanilla RNN achieved an MAE of 0.1250 °C, and the LSTM achieved an MAE of 0.2981 °C. In the lookback window sweep, where we tested lookbacks of 24, 72, 144, and 288 steps, the Vanilla RNN consistently outperformed both the Moving Average and the LSTM across all windows, with its best performance of 0.1246 °C at a 144-step lookback. The LSTM's performance was consistently worse than the RNN, ranging from 0.2306 °C to 0.5213 °C."

### Sample Examiner Q&A
* **Question 1: Why did the Vanilla RNN outperform the LSTM, which contradicts general theory?**
  * *Answer:* First, the LSTM was evaluated in an under-trained state. Because its validation loss rose after Epoch 1, early stopping restored the weights of Epoch 1, before the model could converge. Second, 1-step forecasting on a highly autocorrelated series is dominated by the most recent values, meaning the RNN's natural bias to weight recent states heavily worked in its favor, while the LSTM suffered from overparameterization and training instability.
* **Question 2: How did the Moving Average baseline perform as lookback increased?**
  * *Answer:* Its performance degraded significantly, with MAE rising from 1.2014 °C at a 24-step lookback to 2.8050 °C at a 288-step lookback. This happens because averaging over a longer window dilutes the highly relevant, recent temperature readings with irrelevant temperatures from 4 hours or 12 hours ago.

---

## Part 8: Conclusion & Limitations
"In conclusion, the project successfully implements time-series forecasting. The Vanilla RNN proved highly efficient, achieving an MAE of 0.1246 °C at a 24-hour lookback. However, our empirical results highlighted a massive contradiction in the notebook's written narrative compared to the actual outputs, alongside bugs in the early stopping mechanism for the LSTM. In the future, I plan to run controlled trials with identical layer depths, fix the early stopping validation dropout discrepancy, and benchmark against a Naive persistence baseline."

### Sample Examiner Q&A
* **Question 1: If you had to fix one critical bug in the training pipeline, what would it be?**
  * *Answer:* I would address the early stopping mechanism. Reverting the LSTM to its Epoch 1 weights due to validation fluctuations prevented the model from training. I would increase the early stopping patience and investigate why the validation loss rose immediately.
* **Question 2: What is the primary lesson learned regarding baseline model selection?**
  * *Answer:* The choice of baseline is critical. The notebook compared neural networks to a weak Simple Moving Average baseline. If we compare them to a Naive persistence baseline ($y_t = y_{t-1}$), which achieves an MAE of 0.1568 °C, we find that the Vanilla RNN's improvement is actually modest, and the LSTM is worse than the naive baseline.

---

## ⚠️ Challenges & Limitations (CRITICAL CALLOUTS)

> [!WARNING]
> **1. Narrative-Data Contradiction**: The presentation script directly addresses the conflict where the notebook claims "LSTM outperforms RNN" while the output cells show the exact opposite.
> **2. Early Stopping Failure**: The LSTM early stopped after 8 epochs and reverted to Epoch 1 weights due to validation loss fluctuations, resulting in evaluating an untrained model.
> **3. Unequal Architecture Depth**: The LSTM model is a 2-layer stacked architecture with dropout and a dense MLP head, whereas the Vanilla RNN is a single-layer model with a direct linear head. This makes the comparison unequal.
