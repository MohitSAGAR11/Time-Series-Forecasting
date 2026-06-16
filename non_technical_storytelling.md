# 📖 The Story of the Temperature Tracker

This document presents the project structure and outcomes in a non-technical, narrative format using simple analogies.

---

## 1. The Goal: Guessing the Near Future
Imagine you live in a town where the weather changes slowly. If someone asks you to predict the temperature in exactly **10 minutes**, the simplest guess you can make is to say: *"It will be about the same as it is right now."* 

Because temperature changes gradually, this simple guess is actually very good. On average, it is only off by about **0.15 degrees**. This is our target to beat.

---

## 2. The Weather Diary
To build a system to forecast the temperature, we studied a detailed weather diary. Every 10 minutes for 7 years, a weather station in Germany recorded details like temperature, humidity, and air pressure. This gave us over 420,000 measurements. 

We split this diary chronologically: we studied the first 70% to learn patterns, used the next 15% to practice our predictions, and saved the final 15% to evaluate our models.

---

## 3. Strategy 1: The Rearview Mirror (Simple Moving Average)
Our first strategy is to look at the average temperature of the last few hours and predict that average for the next step. 

Imagine you are driving a car forward, but you can only look at the average curvature of the road you traveled over the last few miles. If the road is straight, this works fine. But if you hit a sharp turn, you will react too slowly because you are averaging out the recent curves with older ones.

In our tests, if we averaged the last 4 hours of temperature, we were off by **1.2 degrees**. If we averaged the last two days, we were off by **2.8 degrees**. The moving average was too slow to react to daily temperature changes.

---

## 4. Strategy 2: The Notepad (Vanilla RNN)
Our second strategy uses a simple memory system.

Imagine a weather assistant sitting at a desk with a small notepad. Every 10 minutes, the assistant checks the current temperature, reads the note they wrote 10 minutes ago, mixes them together, and writes a new note. 

This assistant works very quickly and is highly responsive to recent changes. Because the notepad is small, old notes are scribbled over, meaning the assistant quickly forgets what happened hours ago.

Surprisingly, this short memory is a benefit. Because the temperature in 10 minutes is closely related to the temperature right now, the assistant's focus on recent history is highly effective. The notepad strategy was off by only **0.12 degrees** on average.

---

## 5. Strategy 3: The Gated Diary (LSTM)
Our final strategy uses a sophisticated memory system.

Imagine an assistant with a structured diary. This diary has three rules, managed by "gates":
1. **The Forget Gate** decides what past info to erase.
2. **The Input Gate** decides what new details to record.
3. **The Output Gate** decides what to report.

This system is designed to track long-term trends, like day-and-night cycles or multi-day weather patterns. 

However, because this diary has so many rules, it is easy to overcomplicate. In our tests, the assistant with the diary got distracted by validation fluctuations early in their training, causing them to stop learning before they could master the patterns. As a result, the diary assistant was off by **0.29 degrees** on average—worse than the simpler notepad.

---

## 6. The Takeaway: Keeping It Simple
Our experiment taught us a valuable lesson: **do not use a complex solution when a simple one works better.** 

For predicting temperature just 10 minutes ahead, a simple notepad (Vanilla RNN) is highly effective, cheap to run, and accurate. The gated diary (LSTM) has potential, but its complexity makes it harder to train and prone to errors on simple tasks.

---

## ⚠️ Challenges & Limitations (CRITICAL CALLOUTS)

> [!WARNING]
> **The Diary Trap (Early Stopping Anomaly)**: The sophisticated "Gated Diary" (LSTM) failed because its validation checks were too strict early on, making it stop training prematurely.
> **The Simple Target**: For 10-minute forecasts, the "notepad" (RNN) was enough. A diary is only useful if we want to predict days in advance or handle complex inputs.
