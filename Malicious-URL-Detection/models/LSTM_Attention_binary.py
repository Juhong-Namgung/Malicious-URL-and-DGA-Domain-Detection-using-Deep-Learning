# Load Libraries
import time
import warnings
from datetime import datetime

import tensorflow as tf
from keras import backend as K
from keras import regularizers
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.layers import Embedding, LSTM
from keras.layers.core import Dense, Dropout, Flatten
from keras.models import Sequential
from keras.optimizers import Adam
from keras_self_attention import SeqSelfAttention
from model_evaluator import Evaluator
from model_preprocessor import Preprocessor

warnings.filterwarnings("ignore")
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.9
K.tensorflow_backend.set_session(tf.Session(config=config))

with tf.device("/GPU:0"):

    def lstm_att(max_len=80, emb_dim=32, max_vocab_len=128, W_reg=regularizers.l2(1e-4)):
        """LSTM with Attention model with the Keras Sequential model"""

        model = Sequential()
        model.add(Embedding(input_dim=max_vocab_len, output_dim=emb_dim, input_length=max_len, W_regularizer=W_reg))
        model.add(Dropout(0.2))
        model.add(LSTM(units=128, return_sequences=True))
        model.add(Dropout(0.5))
        model.add(SeqSelfAttention(attention_activation='relu'))
        model.add(Flatten())
        model.add(Dense(8576, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(1, activation='sigmoid'))

        return model

with tf.device("/GPU:0"):

    # Load data using model preprocessor
    x_train, x_test, y_train, y_test = Preprocessor.load_data_binary()

    # Define Deep Learning Model
    model_name = "LSTM_ATT_BI"
    model = lstm_att()

    # Define early stopping
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=3)
    mc = ModelCheckpoint(filepath='./trained_models/' + model_name+ '.hdf5', monitor='val_loss', mode='min',
                         save_best_only=True, verbose=1)

    ''' Training phrase '''
    epochs = 10
    batch_size = 64
    adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

    model.compile(optimizer=adam, loss='binary_crossentropy',
                  metrics=['accuracy', tf.keras.metrics.BinaryAccuracy(),
                           Evaluator.precision, Evaluator.recall, Evaluator.fmeasure])

    dt_start_train = datetime.now()

    history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.11, callbacks=[es, mc])

    dt_end_train = datetime.now()

    ''' Predict phrase '''
    best_model = lstm_att()
    best_model.load_weights('./trained_models/' + model_name+ '.hdf5')
    best_model.compile(optimizer=adam, loss='binary_crossentropy',
                       metrics=['accuracy', tf.keras.metrics.BinaryAccuracy(),
                                Evaluator.precision, Evaluator.recall, Evaluator.fmeasure])

    dt_start_predict = datetime.now()

    y_pred = best_model.predict(x_test, batch_size=64)

    dt_end_predict = datetime.now()

    # Validation curves
    Evaluator.plot_validation_curves(model_name, history, type='binary')
    Evaluator.print_validation_report(history)

    # Experimental result
    result = best_model.evaluate(x_test, y_test, batch_size=64)
    result_dic = dict(zip(best_model.metrics_names, result))

    print('\nAccuracy: {}\n Binary_Accuracy: {}\n'
          'Precision: {}\nRecall: {}\n F-1Score {}\n'
          .format(result_dic['accuracy'], result_dic['binary_accuracy'],
                  result_dic['precision'], result_dic['recall'], result_dic['fmeasure']))

    Evaluator.calculate_measure_binary(best_model, x_test, y_test)

    # Save confusion matrix
    Evaluator.plot_confusion_matrix(model_name, y_test, y_pred, title='Confusion matrix', normalize=False, classes=[0,1])
    time.sleep(5)
    Evaluator.plot_confusion_matrix(model_name, y_test, y_pred, title='Confusion matrix', normalize=True, classes=[0,1])

    # Print Training and predicting time
    print('Train time: ' + str((dt_end_train - dt_start_train)))
    print('Predict time: ' + str((dt_end_predict - dt_start_predict)))