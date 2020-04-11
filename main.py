import nltk
from sklearn.feature_extraction.text import CountVectorizer
import os


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
    return corpus


def sort_matrix(matrix):
    """Sort matrix"""
    tuples = zip(matrix.col, matrix.data)
    return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)

# Work
corpus = get_corpus(path="dataset\\train\\neg", file_limit=20)
cv = CountVectorizer(stop_words='english')
bag_of_words = cv.fit_transform(corpus)
feature_names = cv.get_feature_names()

# Output
rows, words = bag_of_words.shape
print("rows: %s | words: %s" % (rows, words))
print("Top popular words:")
print("word | weight")
for i in sort_matrix(bag_of_words.tocoo())[:10]:
    print(feature_names[i[0]], i[1])
