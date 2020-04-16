import pickle

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer


class ML:

    def __init__(self):
        self._model = None
        self._vectorizer = CountVectorizer()
        self._train_y = None

    def train(self, dir_trains):
        corpus = load_files(dir_trains, encoding='utf-8')
        X = self._vectorizer.fit_transform(corpus.data)
        self._train_y = corpus.target
        self._model = RandomForestClassifier(n_estimators=18)
        self._model.fit(X=X, y=self._train_y)

    def save_model(self, filename):
        if self._model:
            joblib.dump(self._model, '%s.pkl' % filename, compress=9)
        else:
            raise Exception('Model is empty. Train the model first')

    def save_vectorizer(self, filename):
        if self._vectorizer:
            joblib.dump(self._vectorizer, '%s.vec' % filename, compress=9)
        else:
            raise Exception('Vectorizer is empty. Train the model first')

    def save_train_y(self, filename):
            if self._vectorizer:
                joblib.dump(self._train_y, '%s.wght' % filename, compress=9)
            else:
                raise Exception('Weights is empty. Train the model first')

    def load_vectorizer(self, filename):
        self._vectorizer = joblib.load('%s.vec' % filename)

    def load_model(self, filename):
        self._model = joblib.load('%s.pkl' % filename)

    def load_train_y(self, filename):
        self._train_y = joblib.load('%s.wght' % filename)

    def testing(self, dir_tests):
        if self._model:
            corpus = load_files(dir_tests)
            X = self._vectorizer.transform(corpus.data)
            y = self._model.predict(X=X)
            accuracy = accuracy_score(self._train_y, y)
            return y, accuracy
        raise Exception("Model did't train")


dir_trains = "dataset\\_train"
dir_tests = "dataset\\_test"

ml = ML()
# ml.train(dir_trains)
# ml.save_model('wtf_sm')
# ml.save_vectorizer('wtf_sm')
# ml.save_train_y('wtf_sm')
ml.load_vectorizer('wtf_sm')
ml.load_model('wtf_sm')
ml.load_train_y('wtf_sm')
y, accuracy = ml.testing(dir_tests)

print(accuracy)
