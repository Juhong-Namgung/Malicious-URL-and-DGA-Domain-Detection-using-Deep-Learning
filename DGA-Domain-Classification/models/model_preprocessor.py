import pandas as pd
import numpy as np
from sklearn import model_selection

from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.utils import np_utils
from keras import backend as K


class Preprocessor:
    def __init__(self):
        pass

    @staticmethod
    def load_data(__self__):
        """ Load and pre-process data.

        1) Load data from dir
        2) Tokenizing
        3) Padding

        return train and test data

        """

        # Load data
        data_home = '../data/'
        df = pd.read_csv(data_home + 'dga_label.csv', encoding='ISO-8859-1', sep=',')

        # Tokenizing domain string on character level
        # domain string to vector
        tokenizer = Tokenizer(filters='', lower=True, char_level=True)
        tokenizer.fit_on_texts(df.domain)
        url_int_tokens = tokenizer.texts_to_sequences(df.domain)

        # Padding domain integer max_len=64
        # 최대길이 67로 지정
        max_len = 67

        x = sequence.pad_sequences(url_int_tokens, maxlen=max_len)
        y = np.array(df['class'])

        # Cross-validation
        x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.1, random_state=33)

        # DGA class: 0~20: 21개
        y_train_category = np_utils.to_categorical(y_train, 21)
        y_test_category = np_utils.to_categorical(y_test, 21)

        return x_train, x_test, y_train_category, y_test_category

    '''
    Training Metrics
    
    1) precision
    2) recall
    3) F1-score
        
    '''

    @staticmethod
    def precision(self, y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    @staticmethod
    def recall(self, y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    @staticmethod
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

    @staticmethod
    def fmeasure(self, y_true, y_pred):
        return Preprocessor.fbeta_score(self, y_true, y_pred, beta=1)