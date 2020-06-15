from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score
from sklearn.preprocessing import MultiLabelBinarizer

import movie_preprocess


class Model:
    def __init__(self, target_var):
        self.model = MultiLabelBinarizer()
        self.tfidf_vectorizer = TfidfVectorizer()
        self.clf = OneVsRestClassifier(SGDClassifier(loss="log", penalty='l2', class_weight="balanced"))
        self.clf2 = OneVsRestClassifier(SGDClassifier(loss="hinge", penalty='l2', class_weight="balanced"))
        self.fitted = False
        self.model.fit(target_var)
        self.target_var = self.model.transform(target_var)

    def fit(self, xtrain, xval, ytrain, yval):
        # create TF-IDF features
        xtrain_tfidf = self.tfidf_vectorizer.fit_transform(xtrain)
        xval_tfidf = self.tfidf_vectorizer.transform(xval)
        # fit model on train data
        self.clf.fit(xtrain_tfidf, ytrain)
        self.clf2.fit(xtrain_tfidf, ytrain)
        # make predictions for validation set
        y_pred = self.clf.predict(xval_tfidf)
        y_pred2 = self.clf2.predict(xval_tfidf)
        # evaluate performance
        print(f"F1 score micro : {f1_score(yval, y_pred, average='micro')}")
        print(f"F1 score macro : {f1_score(yval, y_pred, average='macro')}")
        print(f"F1 score 2 micro : {f1_score(yval, y_pred2, average='micro')}")
        print(f"F1 score 2 macro : {f1_score(yval, y_pred2, average='macro')}")
        self.fitted = True

    def infer(self, data):
        if not self.fitted:
            raise AssertionError('Should fit data before infer on a model')
        data = movie_preprocess.clean_text(data)
        data = movie_preprocess.remove_stopwords(data)
        vec = self.tfidf_vectorizer.transform([data])
        q_pred = self.clf.predict(vec)
        return self.model.inverse_transform(q_pred)

    def get_target_var(self):
        return self.target_var
