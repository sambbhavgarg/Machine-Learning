import nltk
import random
#from nltk.corpus import movie_reviews
from nltk.classify.scikitlearn import SklearnClassifier
import pickle
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from nltk.classify import ClassifierI
from statistics import mode
from nltk.tokenize import word_tokenize


class VoteClassifier(ClassifierI):
    def __init__(self, *distribs):
        self._distribs = distribs

    def classify(self, feats):
        votes = []
        for c in self._distribs:
            v = c.classify(feats)
            votes.append(v)
        return mode(votes)

    def confidence(self, feats):
        votes = []
        for c in self._distribs:
            v = c.classify(feats)
            votes.append(v)

        choice_votes = votes.count(mode(votes))
        conf = choice_votes / len(votes)
        return conf


docs_f = open("Document/docs.pickle", "rb")
docs = pickle.load(docs_f)
docs_f.close()




w_feats5k_f = open("Document/w_feats5k.pickle", "rb")
w_feats = pickle.load(w_feats5k_f)
w_feats5k_f.close()


def find_feats(document):
    words = word_tokenize(document)
    feats = {}
    for w in w_feats:
        feats[w] = (w in words)

    return feats


featsets = [(find_feats(rev), category) for (rev, category) in docs]

random.shuffle(featsets)
print(len(featsets))

testd = featsets[10000:]
traind = featsets[:10000]



open_file = open("Document/originalnaivebayes5k.pickle", "rb")
distrib = pickle.load(open_file)
open_file.close()


open_file = open("Document/MNB_distrib5k.pickle", "rb")
MNB_distrib = pickle.load(open_file)
open_file.close()



open_file = open("Document/BernoulliNB_distrib5k.pickle", "rb")
BernoulliNB_distrib = pickle.load(open_file)
open_file.close()


open_file = open("Document/LogisticRegression_distrib5k.pickle", "rb")
LogisticRegression_distrib = pickle.load(open_file)
open_file.close()


open_file = open("Document/LinearSVC_distrib5k.pickle", "rb")
LinearSVC_distrib = pickle.load(open_file)
open_file.close()


open_file = open("Document/SGDC_distrib5k.pickle", "rb")
SGDC_distrib = pickle.load(open_file)
open_file.close()




voted_distrib = VoteClassifier(
                                  distrib,
                                  LinearSVC_distrib,
                                  MNB_distrib,
                                  BernoulliNB_distrib,
                                  LogisticRegression_distrib
                )




def sentiment(text):
    feats = find_feats(text)
    return voted_distrib.classify(feats),voted_distrib.confidence(feats)
