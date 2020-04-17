import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import TfidfVectorizer


class ML:

    def __init__(self):
        self._model = None
        self._vectorizer = None

    def save_model(self, filename):
        self._save_file(self._model, filename, 'pkl', 'Model')
        return self

    def save_vectorizer(self, filename):
        self._save_file(self._vectorizer, filename, 'vec', 'Vectorizer')
        return self

    def load_vectorizer(self, filename):
        self._vectorizer = joblib.load('%s.vec' % filename)
        return self

    def load_model(self, filename):
        self._model = joblib.load('%s.pkl' % filename)
        return self

    def _save_file(self, variable, filename, type, err_name: str, compress=9):
        if variable is not None:
            joblib.dump(variable, '%s.%s' % (filename, type), compress=compress)
        else:
            raise Exception('%s is empty. Train the model first' % err_name)

    def train(self, dir_trains, max_iter=1e5, C=5):
        self._vectorizer = TfidfVectorizer(stop_words='english')
        self._model = LogisticRegression(C=C, max_iter=max_iter)
        corpus = load_files(dir_trains, encoding='utf-8')
        X, y = self._vectorizer.fit_transform(corpus.data), corpus.target
        self._model.fit(X=X, y=y)
        return self

    def test(self, dir_tests):
        if self._model:
            corpus = load_files(dir_tests)
            X, y = self._vectorizer.transform(corpus.data), corpus.target
            predict = self._model.predict(X=X)
            return accuracy_score(y, predict)
        raise Exception("Model did't train")


if __name__ == '__main__':
    dir_trains = "dataset\\train"
    dir_tests = "dataset\\test"
    file_name = 'wtf'

    ml = ML()
    ml.train(dir_trains)
    # ml.save_vectorizer(file_name)
    # ml.save_model(file_name)
    accuracy = ml.test(dir_tests)
    print(accuracy)
