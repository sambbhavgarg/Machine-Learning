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



class VoteDistrib(ClassifierI):
    def __init__(self, *distribs):
        self._distribs = distribs

    def classify(self, characs):
        votes = []
        for c in self._distribs:
            v = c.classify(characs)
            votes.append(v)
        return mode(votes)

    def confidence(self, characs):
        votes = []
        for c in self._distribs:
            v = c.classify(characs)
            votes.append(v)

        choice_votes = votes.count(mode(votes))
        conf = choice_votes / len(votes)
        return conf

short_pos = open("/home/sambhavgarrg/Avik-ML100/Mine/TwitPap/positive.txt","r").read()
short_neg = open("/home/sambhavgarrg/Avik-ML100/Mine/TwitPap/negative.txt","r").read()

# move this up here
all_words = []
documents = []


#  j is adject, r is adverb, and v is verb
#allowed_word_types = ["J","R","V"]
allowed_word_types = ["J"]

for p in short_pos.split('\n'):
    documents.append( (p, "pos") )
    words = word_tokenize(p)
    pos = nltk.pos_tag(words)
    for w in pos:
        if w[1][0] in allowed_word_types:
            all_words.append(w[0].lower())


for p in short_neg.split('\n'):
    documents.append( (p, "neg") )
    words = word_tokenize(p)
    pos = nltk.pos_tag(words)
    for w in pos:
        if w[1][0] in allowed_word_types:
            all_words.append(w[0].lower())



save_documents = open("/home/sambhavgarrg/Avik-ML100/Mine/TwitPap/pickled_algos/documents.pickle","wb")
pickle.dump(documents, save_documents)
save_documents.close()


all_words = nltk.FreqDist(all_words)


word_characs = list(all_words.keys())[:5000]


save_word_characs = open("/home/sambhavgarrg/Avik-ML100/Mine/TwitPap/pickled_algos/word_characs5k.pickle","wb")
pickle.dump(word_characs, save_word_characs)
save_word_characs.close()


def find_characs(document):
    words = word_tokenize(document)
    characs = {}
    for w in word_characs:
        characs[w] = (w in words)

    return characs

characsets = [(find_characs(rev), category) for (rev, category) in documents]
        
traind = characsets[:1900]
testd =  characsets[1900:]


distrib_f = open("naivebayes.pickle","rb")
distrib = pickle.load(distrib_f)
distrib_f.close()


distrib = nltk.NaiveBayesClassifier.train(traind)
print("Original Naive Bayes Algo accuracy percent:", (nltk.classify.accuracy(distrib, testd))*100)

distrib.show_most_informative_characs(15)


store_distrib = open("/home/sambhavgarrg/Avik-ML100/Mine/TwitPap/pickled_algos/originalnaivebayes5k.pickle","wb")
pickle.dump(distrib, store_distrib)
store_distrib.close()

MNB_distrib = SklearnClassifier(MultinomialNB())
MNB_distrib.train(traind)
print("MNB_distrib accuracy percent:", (nltk.classify.accuracy(MNB_distrib, testd))*100)

store_distrib = open("/home/sambhavgarrg/Avik-ML100/Mine/TwitPap/pickled_algos/MNB_distrib5k.pickle","wb")
pickle.dump(MNB_distrib, store_distrib)
store_distrib.close()

BernoulliNB_distrib = SklearnClassifier(BernoulliNB())
BernoulliNB_distrib.train(traind)
print("BernoulliNB_distrib accuracy percent:", (nltk.classify.accuracy(BernoulliNB_distrib, testd))*100)

store_distrib = open("/home/sambhavgarrg/Avik-ML100/Mine/TwitPap/pickled_algos/BernoulliNB_distrib5k.pickle","wb")
pickle.dump(BernoulliNB_distrib, store_distrib)
store_distrib.close()

LogisticRegression_distrib = SklearnClassifier(LogisticRegression())
LogisticRegression_distrib.train(traind)
print("LogisticRegression_distrib accuracy percent:", (nltk.classify.accuracy(LogisticRegression_distrib, testd))*100)

store_distrib = open("/home/sambhavgarrg/Avik-ML100/Mine/TwitPap/pickled_algos/LogisticRegression_distrib5k.pickle","wb")
pickle.dump(LogisticRegression_distrib, store_distrib)
store_distrib.close()


LinearSVC_distrib = SklearnClassifier(LinearSVC())
LinearSVC_distrib.train(traind)
print("LinearSVC_distrib accuracy percent:", (nltk.classify.accuracy(LinearSVC_distrib, testd))*100)

store_distrib = open("/home/sambhavgarrg/Avik-ML100/Mine/TwitPap/pickled_algos/LinearSVC_distrib5k.pickle","wb")
pickle.dump(LinearSVC_distrib, store_distrib)
store_distrib.close()

SGDC_distrib = SklearnClassifier(SGDClassifier())
SGDC_distrib.train(traind)
print("SGDClassifier accuracy percent:",nltk.classify.accuracy(SGDC_distrib, testd)*100)

store_distrib = open("/home/sambhavgarrg/Avik-ML100/Mine/TwitPap/pickled_algos/SGDC_distrib5k.pickle","wb")
pickle.dump(SGDC_distrib, store_distrib)
store_distrib.close()
