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
        return self

    def save_model(self, filename):
        self._save_file(self._model, filename, 'Model')
        return self

    def save_vectorizer(self, filename):
        self._save_file(self._vectorizer, filename, 'Vectorizer')
        return self

    def save_train_y(self, filename):
        self._save_file(self._train_y, filename, 'Weight')
        return self

    def load_vectorizer(self, filename):
        self._vectorizer = joblib.load('%s.vec' % filename)
        return self

    def load_model(self, filename):
        self._model = joblib.load('%s.pkl' % filename)
        return self

    def load_train_y(self, filename):
        self._train_y = joblib.load('%s.wght' % filename)
        return self

    def _save_file(self, variable, filename, err_name: str, compress=9):
        if variable is not None:
            joblib.dump(variable, '%s.wght' % filename, compress=compress)
        else:
            raise Exception('%s is empty. Train the model first' % err_name)

    def testing(self, dir_tests):
        if self._model:
            corpus = load_files(dir_tests)
            X = self._vectorizer.transform(corpus.data)
            y = self._model.predict(X=X)
            accuracy = accuracy_score(self._train_y, y)
            return y, accuracy
        raise Exception("Model did't train")


dir_trains = "dataset\\_train"
dir_tests = "dataset\\test"

ml = ML()
# ml.train(dir_trains)
# ml.save_model('wtf_sm').save_vectorizer('wtf_sm').save_train_y('wtf_sm')
ml.load_vectorizer('wtf').load_model('wtf').load_train_y('wtf')
y, accuracy = ml.testing(dir_tests)
print(accuracy)
