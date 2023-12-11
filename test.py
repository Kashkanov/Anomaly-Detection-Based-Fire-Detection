import argparse
import numpy as np
import pandas as pd
import os
from tensorflow.keras.models import load_model
from tabulate import tabulate

def mean_squared_loss(x1, x2):
  diff = x1 - x2
  n_samples = np.prod(diff.shape)
  mean_dist = np.sqrt(np.sum(diff ** 2) / n_samples)
  return mean_dist

def load_and_preprocess_data(filename):
  X_test = np.load(f"{filename}.npy")
  frames = X_test.shape[2]
  frames -= frames % 10
  X_test = X_test[:, :, :frames].reshape(-1, 227, 227, 10, 1)
  return X_test

def detect_anomalies(model, X_test, threshold):
  result = []
  anomalous_count = 0
  for number, bunch in enumerate(X_test):
      t = [number]
      n_bunch = np.expand_dims(bunch, axis=0)
      reconstructed_bunch = model.predict(n_bunch)
      loss = mean_squared_loss(n_bunch, reconstructed_bunch)
      t.append(loss)
      if loss > threshold:
          anomalous_count += 1
          t.append("Anomalous")
      else:
          t.append("Normal")
      result.append(t)
  return result, anomalous_count

def save_results_to_csv(result, filename):
  pd.DataFrame(result, columns=["Bunch No.", "Loss", "Type"]).to_csv(f"{filename}.csv", index_label="Index")

def print_results(result):
  print(tabulate(result, headers=["Bunch No.", "Loss", "Type"], tablefmt="fancy_grid"))

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('filename', type=str)
  args = parser.parse_args()
  model = load_model('Model/AnomalyDetector.h5')
  filename = args.filename
  X_test = load_and_preprocess_data(filename)
  threshold = 0.0003
  result, anomalous_count = detect_anomalies(model, X_test, threshold)
  directory_name = os.path.basename(os.path.normpath(filename))
  savefile = "./results/" + directory_name
  save_results_to_csv(result, savefile)
  print_results(result)
  print(f"Anomalous Events detected: {anomalous_count}")

if __name__ == "__main__":
	main()
