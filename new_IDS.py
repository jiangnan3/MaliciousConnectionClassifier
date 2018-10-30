from keras.models import Sequential
from keras.layers import Dense
from keras.models import load_model
import keras
import pandas
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
import numpy as np
from sklearn.externals import joblib

#--------------Preprocessing-----------------#

data = pandas.read_csv("datacopy", header=None)

feature = data.drop([41], 1)
label = data[41]

scaler = preprocessing.StandardScaler().fit(feature)
joblib.dump(scaler, 'idsscaler.pkl')
#clf = joblib.load('filename.pkl')

feature = scaler.transform(feature)

# --------------Train----------------- #

train_x, test_x, train_y, test_y = train_test_split(feature, label, test_size = 0.3)

ppp = test_y
train_y = to_categorical(train_y, 5)
test_y = to_categorical(test_y, 5)

validation_x = train_x[:20000]
train_x = train_x[20000:]
validation_y = train_y[:20000]
train_y = train_y[20000:]

encoder = load_model("new_encoder.h5")

model = Sequential()
model.add(Dense(9, activation='relu', input_dim=9))
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
print results

model.save("IDS.h5")
