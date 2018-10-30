from keras.layers import Input
from keras.models import Model
from keras.models import Sequential
from keras.layers import Dense
import keras
import pandas
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical

encoderlayer = [5]
classsiferlayer = [5]

data = pandas.read_csv("datacopy", header=None)
simulate_result = open("simulation_result", "w")

for en_layer in encoderlayer:

    feature = data.drop([41], 1)
    label = data[41]

    scaler = preprocessing.StandardScaler().fit(feature)
    feature = scaler.transform(feature)
    feature_train = feature

    validation = feature[:20000]
    feature = feature[20000:]

    encoding_dim = en_layer

    # this is our input placeholder
    input_feature = Input(shape=(feature.shape[1],))
    # "encoded" is the encoded representation of the input
    encoded = Dense(encoding_dim, activation='relu')(input_feature)
    # "decoded" is the lossy reconstruction of the input
    decoded = Dense(feature.shape[1], activation='sigmoid')(encoded)

    # this model maps an input to its reconstruction
    autoencoder = Model(input_feature, decoded)

    # this model maps an input to its encoded representation
    encoder = Model(input_feature, encoded)

    autoencoder.compile(optimizer='adadelta', loss='mean_squared_error')

    autoencoder.fit(feature, feature,
                    epochs=10,
                    batch_size=256,
                    shuffle=True,
                    validation_data=(validation, validation))

    # -----------------------------------------------------------------#
    for cla_layer in classsiferlayer:
        train_x, test_x, train_y, test_y = train_test_split(feature_train, label, test_size=0.3)

        train_y = to_categorical(train_y, 5)
        test_y = to_categorical(test_y, 5)

        validation_x = train_x[:20000]
        train_x = train_x[20000:]
        validation_y = train_y[:20000]
        train_y = train_y[20000:]

        model = Sequential()
        model.add(Dense(cla_layer, activation='relu', input_dim=encoding_dim))
        model.add(Dense(5, activation='softmax'))

        opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)
        model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

        model.fit(encoder.predict(train_x), train_y,
                  epochs=10,
                  batch_size=512,
                  shuffle=True,
                  validation_data=(encoder.predict(validation_x), validation_y))

        results = model.evaluate(encoder.predict(test_x), test_y)
        print "\n"
        print en_layer, cla_layer, results
        recordstring = "\n" + str(en_layer) + "," + str(cla_layer) + ", [" + str(results[0]) + "," + str(
            results[1]) + "] \n"
        print recordstring
        simulate_result.write(recordstring)

simulate_result.close()