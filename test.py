from keras.layers import Input
from keras.models import Model
from keras.models import Sequential
from keras.layers import Dense
import keras
import pandas
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
import numpy as np

odd = []
even = []

for i in range(1000):
    if i % 2 == 0:
        even.append([i] * 10)
    else:
        odd.append([i] * 10)


odd.extend(even)
feature = odd
feature = np.array(feature)

scaler = preprocessing.StandardScaler().fit(feature)
feature = scaler.transform(feature)


oddLabel = [0] * 500
evenLabel = [1] * 500

oddLabel.extend(evenLabel)
label = oddLabel
label = np.asarray(label)


train_x, test_x, train_y, test_y = train_test_split(feature, label, test_size=0.4)

train_y = to_categorical(train_y, 2)
test_y = to_categorical(test_y, 2)

validation_x = train_x[:20]
train_x = train_x[20:]
validation_y = train_y[:20]
train_y = train_y[20:]

model = Sequential()
model.add(Dense(20, activation='sigmoid', input_dim=10))
model.add(Dense(20, activation='sigmoid'))
model.add(Dense(2, activation='softmax'))

opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)
model.compile(optimizer='SGD', loss='mean_squared_error', metrics=['accuracy'])


model.fit(train_x, train_y,
          epochs=10,
          batch_size=124,
          shuffle=True,
          validation_data=(validation_x, validation_y))


result = model.evaluate(test_x, test_y)

print(result)
