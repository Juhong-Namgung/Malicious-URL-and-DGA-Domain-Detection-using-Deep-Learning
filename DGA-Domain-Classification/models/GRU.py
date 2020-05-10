# Load Libraries

import warnings
from datetime import datetime

import tensorflow as tf
from keras import backend as K
from keras import regularizers
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.layers import Embedding, GRU
from keras.layers.core import Dense, Dropout
from keras.models import Sequential
from keras.optimizers import Adam
from model_evaluator import Evaluator
from model_preprocessor import Preprocessor

warnings.filterwarnings("ignore")
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.9
K.tensorflow_backend.set_session(tf.Session(config=config))

with tf.device("/GPU:0"):

    def gru(max_len=67, emb_dim=32, max_vocab_len=39, W_reg=regularizers.l2(1e-4)):
        """GRU model with the Keras Sequential model"""

        model = Sequential()
        model.add(Embedding(input_dim=max_vocab_len, output_dim=emb_dim, input_length=max_len, W_regularizer=W_reg))
        model.add(Dropout(0.5))
        model.add(GRU(128))
        model.add(Dropout(0.5))
        model.add(Dense(21, activation='softmax'))

        return model

with tf.device("/GPU:0"):

    # Load data using model preprocessor
    x_train, x_test, y_train, y_test = Preprocessor.load_data()

    # Define Deep Learning Model
    model_name = "GRU"
    model = gru()

    # Define early stopping
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=3)
    mc = ModelCheckpoint(filepath='./trained_models/' + model_name+ '.hdf5', monitor='val_loss', mode='min',
                         save_best_only=True, verbose=1)

    ''' Training phrase '''
    epochs = 10
    batch_size = 64
    adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

    model.compile(optimizer=adam, loss='categorical_crossentropy',
                  metrics=['accuracy', tf.keras.metrics.CategoricalAccuracy(),
                           Evaluator.precision, Evaluator.recall, Evaluator.fmeasure])

    dt_start_train = datetime.now()

    history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.11, callbacks=[es, mc])

    dt_end_train = datetime.now()

    ''' Predict phrase '''
    best_model = gru()
    best_model.load_weights('./trained_models/' + model_name+ '.hdf5')
    best_model.compile(optimizer=adam, loss='categorical_crossentropy',
                       metrics=['accuracy', tf.keras.metrics.CategoricalAccuracy(),
                                Evaluator.precision, Evaluator.recall, Evaluator.fmeasure])

    dt_start_predict = datetime.now()

    y_pred = best_model.predict(x_test, batch_size=64)

    dt_end_predict = datetime.now()

    # Validation curves
    Evaluator.plot_validation_curves(model_name, history)
    Evaluator.print_validation_report(history)

    # Experimental result
    Evaluator.calculate_measure(best_model, x_test, y_test)

    # Save confusion matrix
    Evaluator.plot_confusion_matrix(model_name, y_test, y_pred, title='Confusion matrix', normalize=True)

    # Print Training and predicting time
    print('Train time: ' + str((dt_end_train - dt_start_train)))
    print('Predict time: ' + str((dt_end_predict - dt_start_predict)))