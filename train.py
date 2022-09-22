from sklearn.neural_network import MLPClassifier

from sklearn.metrics import accuracy_score
from utils import load_data

import os
import pickle

# load RAVDESS dataset
X_train, X_test, y_train, y_test = load_data(test_size=0.30)

# number of samples in training data
print("[+] Number of training samples:", X_train.shape[0])

# number of samples in testing data
print("[+] Number of testing samples:", X_test.shape[0])

# using utils.extract_features() method
print("[+] Number of features:", X_train.shape[1])

#    a grid search algorithmn initialization
model_params = {
    'alpha': 0.01,
    'batch_size': 256,
    'epsilon': 1e-08, 
    'hidden_layer_sizes': (400,), 
    'learning_rate': 'adaptive', 
    'max_iter': 400, 
}
# initialize Multi Layer Perceptron classifier
# with best parameters ( so far )
model = MLPClassifier(**model_params)

# train the model
print("[*] Training the model...")
model.fit(X_train, y_train)

# predict 30% of data to measure how good we are
y_pred = model.predict(X_test)

# calculate the accuracy
accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)

print("Accuracy: {:.1f}%".format(accuracy*100))

# now we save the model
# make result directory if doesn't exist yet
if not os.path.isdir("result"):
    os.mkdir("result")

pickle.dump(model, open("mlp_classifier.model", "wb"))
