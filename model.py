"""
script that provides tools for building and fitting a model
with the tensflow and ares workflow
"""

import tensorflow as tf
import pandas as pd
import numpy as np
import missingno as msno 
from tensorflow.keras import layers
from tensorflow import keras
import matplotlib.pyplot as plt
import missingno as msno
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)

def train_val_test_split(filepath:str, target_col:str="TARGET", impute:bool=False)->pd.DataFrame:
    df = pd.read_csv(filepath)
    print(df.describe())
    print(df.isnull().sum())
    # msno.matrix(df)
    # plt.show()
    if impute:
        impute_func(df, target_col)
    train, val, test = np.split(df.sample(frac=1), [int(0.8*len(df)), int(0.9*len(df))])

    return train, val, test 

def impute_func(df, target_col):
    df = df.copy()
    it = IterativeImputer(max_iter=10, random_state=0)
    str_cols = [x for x in list(df.select_dtypes(np.object_).columns)]
    num_cols = [x for x in list(df.columns) if x not in str_cols and x != target_col]
    
    df.loc[:, num_cols] = it.fit_transform(df.loc[:, num_cols])
    print(df)

def conv_to_dataset(dataframe, target_col="TARGET" , shuffle=True, batch_size=256):
    df = dataframe.copy()
    labels = df.pop(target_col)
    print(df)
    df = {key: value[:,tf.newaxis] for key, value in df.items()}
    ds = tf.data.Dataset.from_tensor_slices((dict(df), labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(dataframe))
    ds = ds.batch(batch_size)
    ds = ds.prefetch(batch_size)
    return ds


def get_normalization_layer(name, dataset):
    # Create a Normalization layer for the feature.
    normalizer = layers.Normalization(axis=None)

    # Prepare a Dataset that only yields the feature.
    feature_ds = dataset.map(lambda x, y: x[name])

    # Learn the statistics of the data.
    normalizer.adapt(feature_ds)

    return normalizer


def get_category_encoding_layer(name, dataset, dtype, max_tokens=None):
    # Create a layer that turns strings into integer indices.
    if dtype == 'string':
        index = layers.StringLookup(max_tokens=max_tokens)
    # Otherwise, create a layer that turns integer values into integer indices.
    else:
        index = layers.IntegerLookup(max_tokens=max_tokens)

    # Prepare a `tf.data.Dataset` that only yields the feature.
    feature_ds = dataset.map(lambda x, y: x[name])

    # Learn the set of possible values and assign them a fixed integer index.
    index.adapt(feature_ds)

    # Encode the integer indices.
    encoder = layers.CategoryEncoding(num_tokens=index.vocabulary_size())

    # Apply multi-hot encoding to the indices. The lambda function captures the
    # layer, so you can use them, or include them in the Keras Functional model later.
    return lambda feature: encoder(index(feature))


train, val, test = train_val_test_split("data/loan_data (1).csv", target_col="TARGET", impute=True)
batch_size = 256
# train_ds = conv_to_dataset(train, batch_size=batch_size)
# val_ds = conv_to_dataset(val, shuffle=False, batch_size=batch_size)
# test_ds = conv_to_dataset(test, shuffle=False, batch_size=batch_size)

# float_cols = [x for x in list(train.select_dtypes(np.float64).columns) if x != "TARGET"]
# int_cols = [x for x in list(train.select_dtypes(np.int64).columns) if x != "TARGET"]
# str_cols = [x for x in list(train.select_dtypes(np.object_).columns)]

# all_inputs = []
# encoded_features = []
# for header in float_cols:
#   numeric_col = tf.keras.Input(shape=(1,), name=header)
#   normalization_layer = get_normalization_layer(header, train_ds)
#   encoded_numeric_col = normalization_layer(numeric_col)
#   all_inputs.append(numeric_col)
#   encoded_features.append(encoded_numeric_col)

# for col, type in zip([int_cols, str_cols], ["int64", "string"]): 
#     for header in col:
#         enc_col = tf.keras.Input(shape=(1,), name=header, dtype=type)
#         enc_cat_layer = get_category_encoding_layer(header, train_ds, type, max_tokens=5)
#         encoded_cat_col = enc_cat_layer(enc_col)
#         all_inputs.append(enc_col)
#         encoded_features.append(encoded_cat_col)

# all_features = tf.keras.layers.concatenate(encoded_features)
# x = tf.keras.layers.Dense(32, activation="relu")(all_features)
# x = tf.keras.layers.Dropout(0.5)(x)
# output = tf.keras.layers.Dense(1)(x)

# model = tf.keras.Model(all_inputs, output)
# model.compile(optimizer='adam',
#               loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
#               metrics=["accuracy", keras.metrics.FalseNegatives(name='fn')])

# tf.keras.utils.plot_model(model, show_shapes=True, rankdir="LR")
# model.fit(train_ds, epochs=10, validation_data=val_ds, verbose=False)

# loss, accuracy, fp = model.evaluate(test_ds)

# print("loss:", loss, "accuracy:", accuracy, "False Positives:", fp)
