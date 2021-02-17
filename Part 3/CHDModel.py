import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras import regularizers

tf.random.set_seed(1234)

def processInput(filename):

  heart_data = pd.read_csv(filename, usecols=range(1, 11))

  heart_features = heart_data.copy()
  heart_labels = heart_features.pop('chd')


  heart_features = np.array(heart_features)

  for row in heart_features:
    if row[4] == 'Present':
      row[4] = 1
    else:
      row[4] = 0

  heart_features = np.asarray(heart_features).astype('float32')

  return heart_features, heart_labels

# Alternative method for splitting heart.csv, that randomizes which
# entries go to training/testing
# Source: https://stackoverflow.com/questions/43697240/how-can-i-split-a-dataset-from-a-csv-file-for-training-and-testing

# heart_data = pd.read_csv("heart.csv")
# heart_data['split'] = np.random.randn(heart_data.shape[0], 1)

# msk = np.random.rand(len(heart_data)) <= 0.80

# heart_train = heart_data[msk]
# heart_test = heart_data[~msk]
# heart_train.to_csv('heart_train.csv', index=False)
# heart_test.to_csv('heart_test.csv', index=False)

x_train, y_train = processInput("heart_train.csv")
x_test, y_test = processInput("heart_test.csv")

normalize = preprocessing.Normalization()
normalize.adapt(x_train)

model = tf.keras.Sequential([
  normalize,
  layers.Dense(512, kernel_regularizer=regularizers.l2(0.001), activation='elu'),
  layers.Dense(512, activation='elu'),
  layers.Dropout(0.3),
  layers.Dense(1)
])

model.compile(loss = tf.losses.MeanSquaredError(), optimizer = tf.optimizers.Adam(), metrics=['accuracy'])

print("--Fit model--")
model.fit(x=x_train, y=y_train, epochs=20, shuffle=True)

# Testing
print("--Evaluate model--")
model_loss, model_acc = model.evaluate(x=x_test, y=y_test, verbose=2)
print(f"Model Loss:    {model_loss:.2f}")
print(f"Model Accuray: {model_acc*100:.1f}%")