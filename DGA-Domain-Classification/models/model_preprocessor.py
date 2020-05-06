import pandas as pd
import numpy as np
import os
import json
from sklearn import model_selection

from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.utils import np_utils
from keras import backend as K
from pathlib import Path

class Preprocessor:
    def __init__(self):
        pass

    # General save model to disk function
    def save_model(model, fileModelJSON, fileWeights):
        if Path(fileModelJSON).is_file():
            os.remove(fileModelJSON)
        json_string = model.to_json()
        with open(fileModelJSON, 'w') as f:
            json.dump(json_string, f)
        if Path(fileWeights).is_file():
            os.remove(fileWeights)
        model.save_weights(fileWeights)

    # Load and preprocess data
    def load_data(__self__):
        # Load data
        DATA_HOME ='../data/'
        df = pd.read_csv(DATA_HOME + 'dga_label.csv',encoding='ISO-8859-1', sep=',')

        # Convert domain string to integer
        # URL 알파벳을 숫자로 변경
        tokenizer = Tokenizer(filters='', lower=True, char_level=True)
        tokenizer.fit_on_texts(df.domain)
        url_int_tokens = tokenizer.texts_to_sequences(df.domain)

        # Padding domain integer max_len=64
        # 최대길이 67로 지정
        max_len = 67

        X = sequence.pad_sequences(url_int_tokens, maxlen=max_len, padding='post')
        y = np.array(df['class'])

        # Cross-validation
        X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.1, random_state=33)

        # DGA class: 0~20: 21개
        y_train_category = np_utils.to_categorical(y_train, 21)
        y_test_category = np_utils.to_categorical(y_test, 21)

        return X_train, X_test, y_train_category, y_test_category

    def precision(self, y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    def recall(self, y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def fbeta_score(self, y_true, y_pred, beta=1):
        if beta < 0:
            raise ValueError('The lowest choosable beta is zero (only precision).')

            # If there are no true positives, fix the F score at 0 like sklearn.
        if K.sum(K.round(K.clip(y_true, 0, 1))) == 0:

            return 0
        p = Preprocessor.precision(self, y_true, y_pred)
        r = Preprocessor.recall(self, y_true, y_pred)
        bb = beta ** 2
        fbeta_score = (1 + bb) * (p * r) / (bb * p + r + K.epsilon())
        return fbeta_score

    def fmeasure(self, y_true, y_pred):
        return Preprocessor.fbeta_score(self, y_true, y_pred, beta=1)

processor = Preprocessor()
processor.load_data()