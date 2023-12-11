import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from model import load_model
import argparse

def parse_arguments():
  parser = argparse.ArgumentParser()
  parser.add_argument('n_epochs', type=int)
  return parser.parse_args()

def preprocess_training_data():
  X_train = np.load('npys/training.npy')
  frames = X_train.shape[2]

  # Ensuring the number of frames is divisible by 10
  frames -= frames % 10
  X_train = X_train[:, :, :frames]
  X_train = X_train.reshape(-1, 227, 227, 10)
  X_train = np.expand_dims(X_train, axis=4)
  Y_train = X_train.copy()

  return X_train, Y_train

def train_model(model, X_train, Y_train, epochs, batch_size):
  # Callbacks for model training
  callback_save = ModelCheckpoint("./Model/AnomalyDetector2.h5", monitor="accuracy", save_best_only=True, mode='max', verbose=1)
  callback_early_stopping = EarlyStopping(monitor='accuracy', patience=3)

  print('Model has been loaded')
  print(model.summary())

  # Model training
  model.fit(X_train, Y_train,
            batch_size=batch_size,
            epochs=epochs,
            callbacks=[callback_save, callback_early_stopping]
            )

def main():
  # Command-line argument parsing
  args = parse_arguments()

  # Loading the model
  model = load_model()

  # Loading and preprocessing training data
  X_train, Y_train = preprocess_training_data()

  # Model training parameters
  epochs = args.n_epochs
  batch_size = 1

  # Train the model
  train_model(model, X_train, Y_train, epochs, batch_size)

if __name__ == "__main__":
  main()
