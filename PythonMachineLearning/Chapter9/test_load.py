import numpy as np

import PythonMachineLearning.Chapter9.load_classifier as load_classifier
import PythonMachineLearning.Chapter9.load_vectorizer as load_vectorizer

clf = load_classifier.clf
vect = load_vectorizer.vect

label = {0: "negative", 1:"positive"}
example = ["I love this movie"]
X = vect.transform(example)
print("Prediction: %s\nProbability: %.2f%%" % (label[clf.predict(X)[0]], np.max(clf.predict_proba(X)) * 100))