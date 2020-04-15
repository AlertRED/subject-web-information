from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import TfidfVectorizer


class ML:

    def __init__(self):
        self._model = None
        self._vectorizer = None
        self._train_y = None

    def train(self, dir_trains):
        corpus = load_files(dir_trains, encoding='utf-8')
        self._vectorizer = TfidfVectorizer()
        X = self._vectorizer.fit_transform(corpus.data)
        self._train_y = corpus.target
        self._model = RandomForestClassifier(n_estimators=18)
        self._model.fit(X=X, y=self._train_y)

    def testing(self, dir_tests):
        corpus = load_files(dir_tests)
        X = self._vectorizer.transform(corpus.data)
        y = self._model.predict(X=X)
        accuracy = accuracy_score(self._train_y, y)
        return y, accuracy


dir_trains = "dataset\\train"
dir_tests = "dataset\\test"

ml = ML()
ml.train("dataset\\_train")
y, accuracy = ml.testing("dataset\\_test")
print(y)
print(accuracy)
