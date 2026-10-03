import json 
import time 

import lightgbm as lgb 
import numpy as np 

import pandas as pd 

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score
)

from sklearn.model_selection import train_test_split

DATA_PATH = "./creditcard.csv"

RANDOM_SEED = 42 

results = {}

start_time = time.perf_counter()

df = pd.read_csv(DATA_PATH)


results["data_load_seconds"] = round(time.perf_counter() - start_time, 4)

X = df.drop(columns=["Class"])
y = df["Class"]
results["split"] = "train=0.6, val=0.15, test=0.25"


results["seed"] = RANDOM_SEED


X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.25, random_state=RANDOM_SEED, stratify=y)

X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2, random_state=RANDOM_SEED, stratify=y_train_val)


model = lgb.LGBMClassifier(random_state=RANDOM_SEED)

model.fit(
    X_train, 
    y_train,
    eval_set=[(X_val, y_val)],
    callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
)

results["training_seconds"] = round(time.perf_counter() - start_time, 4)

# 4: Get best iteration
results["best_iteration"] = int(model.best_iteration_)


#5: Evaluate on test set

y_pred_proba = model.predict_proba(X_test)[:, 1]
y_pred = model.predict(X_test)

# calculate metrics: accuracy, precision, recall, f1, auc_roc
results["auc_roc"] = round(float(roc_auc_score(y_test, y_pred_proba)), 4)
results["accuracy"] = round(float(accuracy_score(y_test, y_pred)), 4)
results["precision"] = round(float(precision_score(y_test, y_pred, average="weighted", zero_division=0)), 4)
results["recall"] = round(float(recall_score(y_test, y_pred, average="weighted", zero_division=0)), 4)
results["f1"] = round(float(f1_score(y_test, y_pred, average="weighted", zero_division=0)), 4)

# 6. Latency measurement: measure the time taken to make predictions on the test set after the model is warmup

single_row = X_test.iloc[[0]]
for _ in range(10):
    model.predict(single_row)

num_iterations = 1000

start_time = time.perf_counter()

for _ in range(num_iterations):
    model.predict(single_row)

latency_seconds = time.perf_counter() - start_time

results["latency_1_row_mili_seconds"] = round((latency_seconds / num_iterations) * 1000, 4)

results["latency_iterations"] = num_iterations

# 7: Throughput measurement: measure the time taken to make predictions on the first 1000 rows of the test set after the model is warmup

batch_dataset = X_test.head(1000)

if len(batch_dataset) < 1000:

    batch_dataset = pd.concat([batch_dataset] * (1000 // len(batch_dataset) + 1), ignore_index=True).head(1000)

start_time = time.perf_counter()

model.predict(batch_dataset)

batch_duration = time.perf_counter() - start_time

results["throughput_1000_rows_per_seconds"] = round(1000 / batch_duration, 2)

with open("results.json", "w", encoding="utf-8") as f:

    json.dump(results, f, indent=4)

print(json.dumps(results, indent=4))




