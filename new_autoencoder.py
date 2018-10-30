from keras.layers import Input, Dense
from keras.models import Model
from keras.layers import Dense
import pandas
from sklearn import preprocessing

data = pandas.read_csv("datacopy", header=None)
feature = data.drop([41], 1)
label = data[41]

# label = (label == 'normal.')
# label = label.astype(int)
# feature = pandas.get_dummies(feature)


scaler = preprocessing.StandardScaler().fit(feature)
feature = scaler.transform(feature)

validation = feature[:20000]
feature = feature[20000:]


encoding_dim = 9

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

#-----------------------------------------------------------------#


autoencoder.fit(feature, feature,
                epochs=10,
                batch_size=256,
                shuffle=True,
                validation_data=(validation, validation))

autoencoder.save("new_autoencoder.h5")
encoder.save("new_encoder.h5")
