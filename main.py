import nltk
from sklearn.feature_extraction.text import CountVectorizer
import os
import re

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


class ML:
    _REPLACE_NO_SPACE = re.compile("[.;:!\'?,\"()\[\]]")
    _REPLACE_WITH_SPACE = re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)")
    _NOT_RUNED_MESSAGE = 'Firstly will use \'run\' method'

    @staticmethod
    def _preprocess_reviews(reviews):
        reviews = [ML._REPLACE_NO_SPACE.sub("", line.lower()) for line in reviews]
        reviews = [ML._REPLACE_WITH_SPACE.sub(" ", line) for line in reviews]
        return reviews

    def _get_corpus(self, path):
        """Return list of sentences from txt-s

        :param path: is path to directory with txt-s
        :param file_limit: how many files need to read (-1 unlimited)
        :return: list of sentences
        """

        corpus = []
        for root, dir, files in os.walk(path, topdown=False):
            for i, file_name in enumerate(files):
                if i == self.files_limit:
                    break
                path = os.path.join(root, file_name)
                with open(path, "r") as file:
                    try:
                        document_text = file.read().lower()
                    except Exception as e:
                        print(file_name, e)
                        continue
                    corpus += nltk.sent_tokenize(document_text)
        return ML._preprocess_reviews(corpus)

    def __init__(self, dir_trains, dir_tests, files_limit=-1):
        self.dir_trains = dir_trains
        self.dir_tests = dir_tests
        self.files_limit = files_limit
        self.feature_to_coef = None
        self.feature_to_coef_sorted = None

    def get_best_positive(self, count=1):
        if self.feature_to_coef_sorted:
            return self.feature_to_coef_sorted[-1:-count - 1:-1]
        raise Exception(ML._NOT_RUNED_MESSAGE)

    def get_best_negative(self, count=1):
        if self.feature_to_coef_sorted:
            return self.feature_to_coef_sorted[:count]
        raise Exception(ML._NOT_RUNED_MESSAGE)

    def get_coef(self, word):
        if self.feature_to_coef:
            return self.feature_to_coef.get(word, None)
        raise Exception(ML._NOT_RUNED_MESSAGE)

    def run(self):
        corpus = self._get_corpus(path=self.dir_trains)
        cv = CountVectorizer(stop_words='english')
        X = cv.fit_transform(corpus)

        count = X.shape[0] // 2
        y = [1] * count + [0] * count

        final_model = LogisticRegression()
        final_model.fit(X=X, y=y)

        self.feature_to_coef = {word: coef for word, coef in
                                sorted(zip(cv.get_feature_names(), final_model.coef_[0]), key=lambda x: x[1])}
        self.feature_to_coef_sorted = sorted(zip(cv.get_feature_names(), final_model.coef_[0]), key=lambda x: x[1])


dir_trains = "dataset\\train\\neg"
dir_tests = "dataset\\test\\neg"
ml = ML(dir_trains, dir_tests, files_limit=20)
print(ml.get_best_positive(2))
ml.run()
print(ml.get_best_positive(2))
print(ml.get_best_negative(2))