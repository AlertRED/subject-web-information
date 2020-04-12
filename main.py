import nltk
from sklearn.feature_extraction.text import CountVectorizer
import os
import re

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


class ML:

    REPLACE_NO_SPACE = re.compile("[.;:!\'?,\"()\[\]]")
    REPLACE_WITH_SPACE = re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)")

    @staticmethod
    def __preprocess_reviews(reviews):
        reviews = [ML.REPLACE_NO_SPACE.sub("", line.lower()) for line in reviews]
        reviews = [ML.REPLACE_WITH_SPACE.sub(" ", line) for line in reviews]
        return reviews

    def __get_corpus(self, path):
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
        return ML.__preprocess_reviews(corpus)

    def __init__(self, dir_trains, dir_tests, count_best, files_limit=-1):
        self.dir_trains = dir_trains
        self.dir_tests = dir_tests
        self.count_best = count_best
        self.files_limit = files_limit

    def test2(self):
        X = []  # put your dates in here
        y = []  # put your kwh in here

        model = LinearRegression()
        model.fit(X, y)

        X_predict = []  # put the dates of which you want to predict kwh here
        y_predict = model.predict(X_predict)

    def test(self):
        train_data = self.__get_corpus(path=self.dir_trains)

        test_data = self.__get_corpus(path=self.dir_tests)

        vectorizer = CountVectorizer()
        output = vectorizer.fit_transform(train_data)
        output_test = vectorizer.transform(test_data)

        l1_logreg = LogisticRegression(penalty="l2")
        l1_logreg.fit(output, train_data)
        print(dir(LogisticRegression))
        out_reduced = l1_logreg.transform(output)
        out_reduced_test = l1_logreg.transform(output_test)

        l2_logreg = LogisticRegression(penalty="l2")
        l2_logreg.fit(out_reduced, test_data)
        pedictions = l2_logreg.predict(out_reduced_test)

    def test3(self):
        corpus = self.__get_corpus(path=self.dir_trains)
        cv = CountVectorizer(stop_words='english')
        X = cv.fit_transform(corpus)

        count = X.shape[0] // 2
        y = [1] * count + [0] * count

        final_model = LogisticRegression()
        final_model.fit(X=X, y=y)

        feature_to_coef = {word: coef for word, coef in zip(cv.get_feature_names(), final_model.coef_[0])}

        print(feature_to_coef['good'])


        # print('Best positive')
        # for best_positive in sorted(feature_to_coef.items(), key=lambda x: x[1], reverse=True)[:self.count_best]:
        #     print(best_positive)
        #
        # print('Best negative')
        # for best_negative in sorted(feature_to_coef.items(), key=lambda x: x[1])[:self.count_best]:
        #     print(best_negative)







    def run(self):
        corpus = self.__get_corpus(path=self.dir_trains)
        cv = CountVectorizer(stop_words='english')
        bag_of_words = cv.fit_transform(corpus)

        corpus = self.__get_corpus(path=self.dir_tests)
        cv = CountVectorizer(stop_words='english')
        test_words = cv.fit_transform(corpus)

        count = bag_of_words.shape[0] // 2
        target = [1] * count + [0] * count
        final_model = LogisticRegression(C=1e4)
        output = final_model.fit(bag_of_words, target)


        print(final_model.predict(test_words))

        # feature_to_coef = {word: coef for word, coef in zip(cv.get_feature_names(), final_model.coef_[0])}
        #
        # print('Best positive')
        # for best_positive in sorted(feature_to_coef.items(), key=lambda x: x[1], reverse=True)[:self.count_best]:
        #     print(best_positive)
        #
        # print('Best negative')
        # for best_negative in sorted(feature_to_coef.items(), key=lambda x: x[1])[:self.count_best]:
        #     print(best_negative)
        #
        # print(final_model.coef_)
        # print(final_model.intercept_)

dir_trains="dataset\\train\\neg"
dir_tests="dataset\\test\\neg"
ml = ML(dir_trains, dir_tests, count_best=10, files_limit=20)
ml.test3()
