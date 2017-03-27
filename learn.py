import matplotlib.pyplot as plt
from skimage import io, img_as_float
from os import listdir
import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import ShuffleSplit
from sklearn.neural_network import MLPClassifier
from utils import degrade_image
from utils import framate_image
from constants import DM
from sklearn.ensemble import RandomForestClassifier

from utils import plot_confusion_matrix
from utils import plot_learning_curve

train = pd.read_csv("deatures.csv")
print(train.describe())

predictors = np.arange(1, DM * DM + 1, 1)

print("------- Learn --------")

polynomial_features = PolynomialFeatures(degree=1, include_bias=False)
#alg = linear_model.LogisticRegression()
# alg = AdaBoostClassifier()
alg = RandomForestClassifier(n_estimators=100)
# alg = SVC()
#alg = MLPClassifier(hidden_layer_sizes=(1000, 1000, 1000))

pipeline = Pipeline([("polynomial_features", polynomial_features),
                     ("logistic_regression", alg)])
scores = cross_val_score(
    pipeline,
    train[predictors],
    train["s"],
    cv=ShuffleSplit(n_splits=10, test_size=0.3, random_state=50)
)
print(scores)
print(scores.mean())

print("Alg")
print(alg)

print("------- Bias-Variance --------")

from sklearn.metrics import confusion_matrix

# Train the algorithm using all the training data

alg.fit(train[predictors], train["s"])
# cnf_matrix = confusion_matrix(train["s"], alg.predict(train[predictors]))
# plot_confusion_matrix(cnf_matrix, classes=[], title='Confusion matrix, without normalization')
# plt.show()

print("------- Bias-Variance --------")

print("Plot")
plot_learning_curve(pipeline, "sdf", train[predictors], train["s"], (-0.1, 1.1), cv=ShuffleSplit(n_splits=10, test_size=0.3, random_state=50), n_jobs=1)
plt.show()

print("-------TEST--------")

for f in listdir("test"):
    degraded = degrade_image('test/' + f, DM)
    io.imsave('test_temp/' + f, degraded)
    frame = framate_image(degraded, DM, -1)

    predictions = alg.predict(frame[predictors])
    print("[TEST] Prediction for file " + str(f) + " = " + str(predictions))