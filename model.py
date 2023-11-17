"""
script that provides tools for building and fitting a model
with the tensflow and kares workflow
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
import os
import tempfile
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTENC


pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)


def train_val_test_split(filepath:str, target_col:str="TARGET", impute:bool=False, resample:bool=False)->pd.DataFrame:
    df = pd.read_csv(filepath, index_col=["SK_ID_CURR"])
    print(df.describe())
    print(df.isnull().sum())
    # msno.matrix(df)
    # plt.show()

    if impute:
        imp_path = "data/df_imputed.csv"
        if os.path.exists(imp_path):
            df = pd.read_csv(imp_path)
            
        else: 
            df = impute_func(df=df, target_col=target_col)
            df.to_csv(imp_path, index=False)
    
    if resample: 
        df = resample_func(df=df, target_col=target_col)
    
    print("Nulls after imputation:")
    print(df.isnull().sum())
    print(df.describe())
    neg, pos = np.bincount(df[target_col])
    train, val, test = np.split(df.sample(frac=1), [int(0.8*len(df)), int(0.9*len(df))])

    return train, val, test, neg, pos


def impute_func(df:pd.DataFrame, target_col:str)->pd.DataFrame:
    df = df.copy()
    str_cols = [x for x in list(df.select_dtypes(np.object_).columns)]
    num_cols = [x for x in list(df.columns) if x not in str_cols and x != target_col]
    # Create an iterative imputer that has min values based on each feature 
    it = IterativeImputer(max_iter=10, random_state=0, min_value=[min for min in list(df[num_cols].min())])
    df.loc[:, num_cols] = it.fit_transform(df.loc[:, num_cols])
    df.loc[:, str_cols] = df.loc[:, str_cols].apply(lambda x: x.fillna(x.value_counts().index[0]))

    return df


def resample_func(df:pd.DataFrame, target_col:str):
    df = df.copy()
    y = df.pop(target_col)
    X = df
    str_cols = [x for x in list(df.select_dtypes(np.object_).columns)]
    oversample = SMOTENC(sampling_strategy=0.1, categorical_features=str_cols)
    under = RandomUnderSampler(sampling_strategy=0.5)
    X, y = oversample.fit_resample(X, y)
    X, y = under.fit_resample(X, y)
    y = y.to_frame()
    X['tmp'] = [i for i in range(len(X))]
    y['tmp'] = [i for i in range(len(y))]
    print(X, y)
    df = pd.merge(X, y)
    df = df.drop(columns=["tmp"])
    print(df.columns)
    return df


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


def build_mod(metrics:list, encoded_features:list, all_inputs:list, optimizer:str='adam' ,output_bias=None):
    if output_bias is not None:
        output_bias = tf.keras.initializers.Constant(output_bias)
    all_features = tf.keras.layers.concatenate(encoded_features)
    x = tf.keras.layers.Dense(32, activation="relu")(all_features)
    x = tf.keras.layers.Dropout(0.5)(x)
    output = tf.keras.layers.Dense(1, activation='sigmoid', bias_initializer=output_bias)(x)
    model = tf.keras.Model(all_inputs, output)
    model.compile(optimizer=optimizer,
                loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                metrics=metrics)

    return model
  

train, val, test, neg, pos = train_val_test_split("data/loan_data (1).csv", target_col="TARGET", impute=True, resample=True)
total = neg + pos
print('Examples:\n    Total: {}\n    Positive: {} ({:.2f}% of total)\n'.format(
    total, pos, 100 * pos / total))


batch_size = 2046
train_ds = conv_to_dataset(train, shuffle=False ,batch_size=batch_size)
val_ds = conv_to_dataset(val, shuffle=False, batch_size=batch_size)
test_ds = conv_to_dataset(test, shuffle=False, batch_size=batch_size)
model_path = 'housing_model.keras'

metrics = [
        keras.metrics.BinaryCrossentropy(name='cross entropy'),  # same as model's loss
        keras.metrics.MeanSquaredError(name='Brier score'),
        keras.metrics.TruePositives(name='tp'),
        keras.metrics.FalsePositives(name='fp'),
        keras.metrics.TrueNegatives(name='tn'),
        keras.metrics.FalseNegatives(name='fn'), 
        keras.metrics.BinaryAccuracy(name='accuracy'),
        keras.metrics.Precision(name='precision'),
        keras.metrics.Recall(name='recall'),
        keras.metrics.AUC(name='auc'),
        keras.metrics.AUC(name='prc', curve='PR'), # precision-recall curve
    ]

if os.path.exists(model_path):
    model = tf.keras.models.load_model(model_path)

else:
    float_cols = [x for x in list(train.select_dtypes(np.float64).columns) if x != "TARGET"]
    int_cols = [x for x in list(train.select_dtypes(np.int64).columns) if x != "TARGET"]
    str_cols = [x for x in list(train.select_dtypes(np.object_).columns)]

    all_inputs = []
    encoded_features = []
    for header in float_cols:
        numeric_col = tf.keras.Input(shape=(1,), name=header)
        normalization_layer = get_normalization_layer(header, train_ds)
        encoded_numeric_col = normalization_layer(numeric_col)
        all_inputs.append(numeric_col)
        encoded_features.append(encoded_numeric_col)

    for col, type in zip([int_cols, str_cols], ["int64", "string"]): 
        for header in col:
            enc_col = tf.keras.Input(shape=(1,), name=header, dtype=type)
            enc_cat_layer = get_category_encoding_layer(header, train_ds, type, max_tokens=5)
            encoded_cat_col = enc_cat_layer(enc_col)
            all_inputs.append(enc_col)
            encoded_features.append(encoded_cat_col)


    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_prc', 
        verbose=1,
        patience=10,
        mode='max',
        restore_best_weights=True)

    init_bias = np.log([pos/neg])
    initial_weights = os.path.join(tempfile.mkdtemp(), 'initial_weights')
    

    model = build_mod(metrics=metrics, encoded_features=encoded_features, all_inputs=all_inputs, output_bias=init_bias)
    model.save_weights(initial_weights)
        
    model.fit(train_ds, epochs=10, validation_data=val_ds, verbose=1, callbacks=[early_stopping])
    model.save(model_path)
