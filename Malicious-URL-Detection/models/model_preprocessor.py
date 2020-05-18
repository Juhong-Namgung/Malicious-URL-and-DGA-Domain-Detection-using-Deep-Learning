import pandas as pd
import numpy as np
from sklearn import model_selection
from string import printable

from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.utils import np_utils


class Preprocessor:
    def __init__(self):
        pass

    @staticmethod
    def load_data_binary():
        """ Load and pre-process data.

        1) Load data from dir
        2) Tokenizing
        3) Padding

        return train and test data

        """

        # Load data
        data_home = '../data/'
        df = pd.read_csv(data_home + 'url_label.csv', encoding='ISO-8859-1', sep=',')

        # Tokenizing domain string on character level
        # domain string to vector
        url_int_tokens = [[printable.index(x) + 1 for x in url if x in printable] for url in df.url]

        # tokenizer = Tokenizer(filters='', lower=False, char_level=True)
        # tokenizer.fit_on_texts(df.url)
        #
        # print(tokenizer.word_index)
        #
        # url_int_tokens = tokenizer.texts_to_sequences(df.url)

        # Padding domain integer max_len=64
        # 최대길이 80으로 지정
        max_len = 80

        x = sequence.pad_sequences(url_int_tokens, maxlen=max_len)

        label_arr = []
        for i in df['class']:
            if i == 0:
                label_arr.append(0)
            else :
                label_arr.append(1)

        y = np.array(label_arr)

        # Cross-validation
        x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.1, random_state=33)

        return x_train, x_test, y_train, y_test

    @staticmethod
    def load_data_multi():
        """ Load and pre-process data.

        1) Load data from dir
        2) Tokenizing
        3) Padding

        return train and test data

        """

        # Load data
        data_home = '../data/'
        df = pd.read_csv(data_home + 'url_label.csv', encoding='ISO-8859-1', sep=',')

        # Tokenizing domain string on character level
        # domain string to vector
        url_int_tokens = [[printable.index(x) + 1 for x in url if x in printable] for url in df.url]

        # tokenizer = Tokenizer(filters='', lower=False, char_level=True)
        # tokenizer.fit_on_texts(df.url)
        #
        # print(tokenizer.word_index)
        #
        # url_int_tokens = tokenizer.texts_to_sequences(df.url)

        # Padding domain integer max_len=64
        # 최대길이 80으로 지정
        max_len = 80

        x = sequence.pad_sequences(url_int_tokens, maxlen=max_len)
        y = np.array(df['class'])

        # Cross-validation
        x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.1, random_state=33)

        # URL class: 0~3: 4개
        y_train_category = np_utils.to_categorical(y_train, 4)
        y_test_category = np_utils.to_categorical(y_test, 4)

        return x_train, x_test, y_train_category, y_test_category
