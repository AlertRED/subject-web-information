import nltk
from sklearn.feature_extraction.text import CountVectorizer
import os
import re

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

REPLACE_NO_SPACE = re.compile("[.;:!\'?,\"()\[\]]")
REPLACE_WITH_SPACE = re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)")


def preprocess_reviews(reviews):
    reviews = [REPLACE_NO_SPACE.sub("", line.lower()) for line in reviews]
    reviews = [REPLACE_WITH_SPACE.sub(" ", line) for line in reviews]
    return reviews


def get_corpus(path, file_limit=-1):
    """Return list of sentences from txt-s

    :param path: is path to directory with txt-s
    :param file_limit: how many files need to read (-1 unlimited)
    :return: list of sentences
    """

    corpus = []
    for root, dir, files in os.walk(path, topdown=False):
        for i, file_name in enumerate(files):
            if i == file_limit:
                break
            path = os.path.join(root, file_name)
            with open(path, "r") as file:
                try:
                    document_text = file.read().lower()
                except Exception as e:
                    print(file_name, e)
                    continue
                corpus += nltk.sent_tokenize(document_text)
    return preprocess_reviews(corpus)


def sort_matrix(matrix):
    """Sort matrix"""
    tuples = zip(matrix.col, matrix.data)
    return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)


# Work
corpus = get_corpus(path="dataset\\train\\neg", file_limit=20)
cv = CountVectorizer(stop_words='english')
bag_of_words = cv.fit_transform(corpus)

corpus = get_corpus(path="dataset\\test\\neg", file_limit=20)
cv = CountVectorizer(stop_words='english')
test_words = cv.fit_transform(corpus)

count = bag_of_words.shape[0] // 2
target = [1] * count + [0] * count
final_model = LogisticRegression(C=0.5)
final_model.fit(bag_of_words, target)
print(accuracy_score(target, final_model.predict(bag_of_words)))

feature_to_coef = {word: coef for word, coef in zip(cv.get_feature_names(), final_model.coef_[0])}

for best_positive in sorted(feature_to_coef.items(), key=lambda x: x[1], reverse=True)[:5]:
    print(best_positive)

for best_negative in sorted(feature_to_coef.items(), key=lambda x: x[1])[:5]:
    print(best_negative)

# # Output
# rows, words = bag_of_words.shape
# print("rows: %s | words: %s" % (rows, words))
# print("Top popular words:")
# print("word | weight")
# for i in sort_matrix(bag_of_words.tocoo())[:10]:
#     print(feature_names[i[0]], i[1])
