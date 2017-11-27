from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.externals import six
from sklearn.base import clone
from sklearn.pipeline import _name_estimators
import numpy as np
import operator

class MajorityVoteClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, classifiers, vote="classlabel", weights=None):
        self.classifiers = classifiers
        self.named_classifiers = {key: value for key, value in _name_estimators(classifiers)}
        self.vote = vote
        self.weight = weights

    def fit(self, X, y):
        self.labelenc = LabelEncoder()
        self.labelenc.fit(y)
        self.classes = self.labelenc.classes_
        self.classifiers = []
        for clf in self.classifiers:
            fitted_clf = clone(clf).fit(X, self.labelenc.transform(y))
            self.classifiers.append(fitted_clf)
        return self

    """
    # メモ:

    ## idea

    複数のclassifierやらregressorをこんな感じでまとめて、
    最大効率を叩き出す機械学習モデルを一発で投げる的なclassを作る

    複数のclassifierの予測確率*予測をensembleするやつ
    例: 正しい予測をする確率: Logistic 90%, tree 80%, SVM: 85%
    あるデータでのpredict
    Logistic "classB", tree "ClassA", SVM "classA"
    classAの確率 = (0.8 + 0.85)  /3 = .55
    classBの確率 = 0.9  /3 = 0.45
    以上よりclass Aを採択的な感じのClassifierを作る?

    後、ensembleのtree関連はRでしこたまやったのでpass



    """