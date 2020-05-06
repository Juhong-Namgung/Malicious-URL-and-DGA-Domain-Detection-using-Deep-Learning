# Load Libraries
import warnings

import model_evaluator
import model_preproecessor
import tensorflow as tf
from keras import regularizers
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.layers import Input, LSTM, Embedding
from keras.layers.core import Dense, Dropout
from keras.models import Model
from keras.models import load_model
from keras.optimizers import Adam

warnings.filterwarnings("ignore")

with tf.device("/GPU:0"):

    def simple_lstm(max_len=67, emb_dim=32, max_vocab_len=38, lstm_output_size=32, W_reg=regularizers.l2(1e-4)):
        # Input
        main_input = Input(shape=(max_len,), dtype='int32', name='main_input')

        # Embedding layer
        emb = Embedding(input_dim=max_vocab_len, output_dim=emb_dim,
                        input_length=max_len, dropout=0.2, W_regularizer=W_reg)(main_input)

        # LSTM layer
        lstm = LSTM(lstm_output_size)(emb)
        lstm = Dropout(0.5)(lstm)

        # Output layer (last fully connected layer)
        output = Dense(21, activation='softmax', name='output')(lstm)

        # Compile model and define optimizer
        model = Model(input=[main_input], output=[output])
        adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        model.compile(optimizer=adam, loss='categorical_crossentropy',  metrics=['accuracy', tf.keras.metrics.CategoricalAccuracy(), preprocess.fmeasure, preprocess.recall, preprocess.precision])

        return model

with tf.device("/GPU:0"):
    epochs = 10
    batch_size = 64

    # Load data using model preprocessor
    preprocess = model_preproecessor.Preprocessor()

    X_train, X_test, y_train, y_test = preprocess.load_data()

    # define CNN model
    model_name = "LSTM"
    model = simple_lstm()

    # Define early stopping
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=3)
    mc = ModelCheckpoint('./trained_models/' + model_name+ '.h5', monitor='val_loss', mode='min', save_best_only=True)

    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.11, callbacks=[es, mc])

    saved_model = load_model('./trained_models/' + model_name+ '.h5', compile=False)
    y_pred = saved_model.predict(X_test, batch_size=64)

    evaluator = model_evaluator.Evaluator()

    # Validation curves
    evaluator.plot_validation_curves(model_name, history)
    evaluator.print_validation_report(history)

    # Experimental result
    evaluator.calculate_measrue(saved_model, X_test, y_test)

    # model.summary()

    # Save confusion matrix
    evaluator.plot_confusion_matrix(model_name, y_test, y_pred, title='Confusion matrix', normalize=True)

    # Save final training model
    # save_model("../models/" + model_name + ".json", "../models/" + model_name + ".h5")
