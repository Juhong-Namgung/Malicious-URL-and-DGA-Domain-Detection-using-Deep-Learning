# Load Libraries
import warnings
from datetime import datetime

import tensorflow as tf
from keras import backend as K
from keras import regularizers
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.layers import Input, ELU, Embedding, BatchNormalization, Convolution1D, GlobalMaxPooling1D, concatenate
from keras.layers.core import Dense, Dropout
from keras.models import Model
from keras.optimizers import Adam
from model_evaluator import Evaluator
from model_preprocessor import Preprocessor

warnings.filterwarnings("ignore")
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.9
K.tensorflow_backend.set_session(tf.Session(config=config))

with tf.device("/GPU:0"):

    def conv_fully(max_len=67, emb_dim=32, max_vocab_len=39, W_reg=regularizers.l2(1e-4)):
        """CNN model with the Keras functional API"""

        # Input
        main_input = Input(shape=(max_len,), dtype='int32', name='main_input')

        # Embedding layer
        emb = Embedding(input_dim=max_vocab_len, output_dim=emb_dim, input_length=max_len, W_regularizer=W_reg)(main_input)
        emb = Dropout(0.5)(emb)

        def get_conv_layer(emb, kernel_size=5, filters=256):
            # Conv layer
            conv = Convolution1D(kernel_size=kernel_size, filters=filters, border_mode='same')(emb)
            conv = ELU()(conv)
            conv = GlobalMaxPooling1D()(conv)
            conv = Dropout(0.5)(conv)
            return conv

        # Multiple Conv Layers
        # 커널 사이즈를 다르게 한 conv
        conv1 = get_conv_layer(emb, kernel_size=2, filters=256)
        conv2 = get_conv_layer(emb, kernel_size=3, filters=256)
        conv3 = get_conv_layer(emb, kernel_size=4, filters=256)
        conv4 = get_conv_layer(emb, kernel_size=5, filters=256)

        # Fully Connected Layers
        # 위 결과 합침
        merged = concatenate([conv1, conv2, conv3, conv4], axis=1)

        hidden1 = Dense(1024)(merged)
        hidden1 = ELU()(hidden1)
        hidden1 = BatchNormalization(mode=0)(hidden1)
        hidden1 = Dropout(0.5)(hidden1)

        hidden2 = Dense(256)(hidden1)
        hidden2 = ELU()(hidden2)
        hidden2 = BatchNormalization(mode=0)(hidden2)
        hidden2 = Dropout(0.5)(hidden2)

        hidden3 = Dense(64)(hidden2)
        hidden3 = ELU()(hidden3)
        hidden3 = BatchNormalization(mode=0)(hidden3)
        hidden3 = Dropout(0.5)(hidden3)

        # Output layer (last fully connected layer)
        # 마지막 클래스 결정하는 layer
        output = Dense(21, activation='softmax', name='main_output')(hidden3)

        model = Model(input=[main_input], output=[output])

        return model

with tf.device("/GPU:0"):

    # Load data using model preprocessor
    x_train, x_test, y_train, y_test = Preprocessor.load_data()

    # Define Deep Learning Model
    model_name = "1DCNN"
    model = conv_fully()

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
    best_model = conv_fully()
    best_model.load_weights('./trained_models/' + model_name+ '.h5')
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