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

  # Preprocessing
  inputs = {}

  for name, column in heart_features.items():
    dtype = column.dtype
    if dtype == object:
      dtype = tf.string
    else:
      dtype = tf.float32

    inputs[name] = tf.keras.Input(shape=(1,), name=name, dtype=dtype)

  numeric_inputs = {name:input for name, input in inputs.items() if input.dtype==tf.float32}

  x = layers.Concatenate()(list(numeric_inputs.values()))
  norm = preprocessing.Normalization()
  norm.adapt(np.array(heart_data[numeric_inputs.keys()]))
  all_numeric_inputs = norm(x)

  preprocessed_inputs = [all_numeric_inputs]

  for name, input in inputs.items():
    if input.dtype == tf.float32:
      continue

    lookup = preprocessing.StringLookup(vocabulary=np.unique(heart_features[name]))
    one_hot = preprocessing.CategoryEncoding(max_tokens=lookup.vocab_size())

    x = lookup(input)
    x = one_hot(x)
    preprocessed_inputs.append(x)

  preprocessed_inputs_cat = layers.Concatenate()(preprocessed_inputs)

  heart_preprocessing = tf.keras.Model(inputs, preprocessed_inputs_cat)

  heart_features_dict = {name: np.array(value) for name, value in heart_features.items()}

  def heart_model(preprocessing_head, inputs):
    body = tf.keras.Sequential([
      layers.Dense(512, kernel_regularizer=regularizers.l2(0.001), activation='elu'),
      layers.Dense(512, activation='elu'),
      layers.Dropout(0.3),
      layers.Dense(1)
    ])

    preprocessed_inputs = preprocessing_head(inputs)
    result = body(preprocessed_inputs)
    model = tf.keras.Model(inputs, result)

    model.compile(loss=tf.losses.BinaryCrossentropy(from_logits=True), optimizer=tf.optimizers.Adam(), metrics=['accuracy'])
    return model

  heart_model = heart_model(heart_preprocessing, inputs)

  return heart_features_dict, heart_labels, heart_model


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

x_train, y_train, heart_model = processInput("heart_train.csv")
x_test, y_test, model = processInput("heart_test.csv")

print("--Fit model--")
heart_model.fit(x=x_train, y=y_train, epochs=20, shuffle=True)

# Testing
print("--Evaluate model--")
model_loss, model_acc = heart_model.evaluate(x=x_test, y=y_test, verbose=2)
print(f"Model Loss:    {model_loss:.2f}")
print(f"Model Accuray: {model_acc*100:.1f}%")