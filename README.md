# MaliciousConnectionClassifier
Used deep learning tools to classify malicious network connections of KDDCup99 dataset and achieved 99.8% accuracy. The classifier adopts an autoencoder and fully connected neural network, based on Tensorflow and Keras.

numeric_data.py is used to numeric non-numerical features of original dataset.

new_autoencoder.py is used to train the auto encoder.

new_IDS.py is used to train the classifier with the pre-trained auto encoder.

evaluation.py is used to evaluate the model performance of 4 different attack categories.

simulation.py is used to find the best structure of the classifier.

More detailed information of this project, including background, motivation, design, implementation, and performance can be found in the report IDS_Report.pdf.
