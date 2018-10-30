import pandas
from keras.models import load_model
import numpy as np
from sklearn.externals import joblib

scaler = joblib.load('idsscaler.pkl')
ids = load_model("IDS.h5")
encoder = load_model("new_encoder.h5")
data = pandas.read_csv("testdatacopy", header=None)

feature = data.drop([41], 1)
label = data[41]

dosfeature = feature[data[41].isin([0])]
doslabel = label[data[41].isin([0])]
notdosfeature = feature[~data[41].isin([0])]
notdoslabel = np.repeat(doslabel[0:1].values, [notdosfeature.shape[0]], axis=0)

u2rfeature = feature[data[41].isin([1])]
u2rlabel = label[data[41].isin([1])]
notu2rfeature = feature[~data[41].isin([1])]
notu2rlabel = np.repeat(u2rlabel[0:1].values, [notu2rfeature.shape[0]], axis=0)

r21feature = feature[data[41].isin([2])]
r21label = label[data[41].isin([2])]
notr21feature = feature[~data[41].isin([2])]
notr21label = np.repeat(r21label[0:1].values, [notr21feature.shape[0]], axis=0)

probefeature = feature[data[41].isin([3])]
probelabel = label[data[41].isin([3])]
notprobefeature = feature[~data[41].isin([3])]
notprobelabel = np.repeat(probelabel[0:1].values, [notprobefeature.shape[0]], axis=0)

normalfeature = feature[data[41].isin([4])]
normallabel = label[data[41].isin([4])]
notnormalfeature = feature[~data[41].isin([4])]
notnormallabel = np.repeat(normallabel[0:1].values, [notnormalfeature.shape[0]], axis=0)


# ---------------dos------------------ #

dospredict = ids.predict(encoder.predict(scaler.transform(dosfeature)))
x = np.argmax(dospredict, axis=1)
y = ~(x == doslabel)
y = y.astype(int)
print sum(y), y.shape[0]
dosfn = (sum(y)) / float(y.shape[0])
print "dosfn:", dosfn

print "\n"

notdospredict = ids.predict(encoder.predict(scaler.transform(notdosfeature)))
x = np.argmax(notdospredict, axis=1)
y = (x == notdoslabel)
y = y.astype(int)

print (sum(y))/float(y.shape[0])
dosfp = (sum(y))/float(y.shape[0])
print "dosfp:", dosfp

print "------\n\n"

# ---------------u2r------------------ #

u2rpredict = ids.predict(encoder.predict(scaler.transform(u2rfeature)))
x = np.argmax(u2rpredict, axis=1)
y = ~(x == u2rlabel)
y = y.astype(int)
print sum(y), y.shape[0]
u2rfn = (sum(y)) / float(y.shape[0])
print "u2rfn:", u2rfn

notu2rpredict = ids.predict(encoder.predict(scaler.transform(notu2rfeature)))
x = np.argmax(notu2rpredict, axis=1)
y = (x == notu2rlabel)
y = y.astype(int)
print "\n"
print (sum(y))/float(y.shape[0])
u2rfp = (sum(y))/float(y.shape[0])
print "u2rfp:", u2rfp

print "------\n\n"

# ---------------r21------------------ #

r21predict = ids.predict(encoder.predict(scaler.transform(r21feature)))
x = np.argmax(r21predict, axis=1)
y = ~(x == r21label)
y = y.astype(int)
print sum(y), y.shape[0]
r21fn = (sum(y)) / float(y.shape[0])
print "r21fn:", r21fn
print "\n"
notr21predict = ids.predict(encoder.predict(scaler.transform(notr21feature)))
x = np.argmax(notr21predict, axis=1)
y = (x == notr21label)
y = y.astype(int)

print (sum(y))/float(y.shape[0])
r21fp = (sum(y))/float(y.shape[0])
print "r21fp:", r21fp

print "-------\n\n"
# ---------------probe------------------ #

probepredict = ids.predict(encoder.predict(scaler.transform(probefeature)))
x = np.argmax(probepredict, axis=1)
y = ~(x == probelabel)
y = y.astype(int)
print sum(y), y.shape[0]
probefn = (sum(y)) / float(y.shape[0])
print "probefn:", probefn
print "\n"
notprobepredict = ids.predict(encoder.predict(scaler.transform(notprobefeature)))
x = np.argmax(notprobepredict, axis=1)
y = (x == notprobelabel)
y = y.astype(int)

print (sum(y))/float(y.shape[0])
probefp = (sum(y))/float(y.shape[0])
print "probefp:", probefp

print "-----\n\n"
# ---------------normal------------------ #

normalpredict = ids.predict(encoder.predict(scaler.transform(normalfeature)))
x = np.argmax(normalpredict, axis=1)
y = ~(x == normallabel)
y = y.astype(int)
print sum(y), y.shape[0]
normalfn = (sum(y)) / float(y.shape[0])
print "normalfn:", normalfn
print "\n"
notnormalpredict = ids.predict(encoder.predict(scaler.transform(notnormalfeature)))
x = np.argmax(notnormalpredict, axis=1)
y = (x == notnormallabel)
y = y.astype(int)

print (sum(y))/float(y.shape[0])
normalfp = (sum(y))/float(y.shape[0])
print "normalfp:", normalfp

print "------\n\n"