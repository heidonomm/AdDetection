import csv
import re
from os import listdir
from collections import defaultdict
import random

import numpy as np
import nltk
import sklearn

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import roc_auc_score, confusion_matrix


class AdDetection():
    def __init__(self):
        self.k = 2  # how many previous sentences to use for context building
        self.emb_dim = 25
        self.train_valid_split_proportion = 0.8

        self.path = "./ml_interview_ads_data"
        self.embeddings_dict = self.form_embeddings_dict()
        print("Embeddings loaded")
        self.training_files, self.validation_files = self.get_train_valid_files(
            self.path)
        self.train_x, self.train_labels = self.form_training_data()
        print("Training data loaded")
        self.valid_x, self.valid_labels = self.form_valid_data()
        print("Validation data loaded")
        self.imbalance_ratio = self.get_imbalance_ratio()

    def get_train_valid_files(self, path):
        all_files = listdir(path)
        nr_of_training_files = int(
            len(all_files) * self.train_valid_split_proportion)
        train_files = random.sample(all_files, nr_of_training_files)
        valid_files = [file for file in all_files if file not in train_files]
        return train_files, valid_files

    def form_training_data(self):
        train_x = list()
        train_labels = list()
        for file in self.training_files:
            context_queue = list()
            while len(context_queue) != self.k:
                context_queue.append(None)
            with open(self.path + f"/{file}", 'r') as in_file:
                reader = csv.reader(in_file)
                for index, row in enumerate(reader):
                    if row[0] == "O":
                        train_labels.append(0)
                    else:
                        train_labels.append(1)

                    current_sentence = row[1].lower()
                    ctx_embedding = self.get_ctx_embedding(
                        context_queue, current_sentence)
                    train_x.append(ctx_embedding)

                    context_queue.pop(0)
                    context_queue.append(current_sentence)

        return train_x, train_labels

    def form_valid_data(self):
        valid_x = list()
        valid_labels = list()
        for file in self.validation_files:
            context_queue = list()
            while len(context_queue) != self.k:
                context_queue.append(None)
            with open(self.path + f"/{file}", 'r') as in_file:
                reader = csv.reader(in_file)
                previous_sentence = None
                for index, row in enumerate(reader):
                    if row[0] == "O":
                        valid_labels.append(0)
                    else:
                        valid_labels.append(1)

                    current_sentence = row[1].lower()
                    ctx_embedding = self.get_ctx_embedding(
                        context_queue, current_sentence)
                    valid_x.append(ctx_embedding)
                    previous_sentence = current_sentence

                    context_queue.pop(0)
                    context_queue.append(current_sentence)

        return valid_x, valid_labels

    def form_test_data(self):
        test_x = list()
        test_labels = list()
        test_files = listdir("./test_data")
        for file in test_files:
            context_queue = list()
            while len(context_queue) != self.k:
                context_queue.append(None)
            with open(f"./test_data/{file}", 'r') as in_file:
                reader = csv.reader(in_file)
                previous_sentence = None
                for index, row in enumerate(reader):
                    if row[0] == "O":
                        test_labels.append(0)
                    else:
                        test_labels.append(1)

                    current_sentence = row[1].lower()
                    ctx_embedding = self.get_ctx_embedding(
                        context_queue, current_sentence)
                    test_x.append(ctx_embedding)
                    previous_sentence = current_sentence

                    context_queue.pop(0)
                    context_queue.append(current_sentence)

        return test_x, test_labels

    def get_ctx_embedding(self, context_queue, new_snt):
        kalkun = new_snt
        """
        get sentence embedding by getting the mean of array of word embeddings
        ctx is the concatenation of previous sent and next sent
        """
        prev_snt_embeds = list()
        for sent in context_queue:
            if sent is None:
                prev_snt_embeds.append(np.zeros(self.emb_dim))
            else:
                snt_embs = list()
                prev_snt = re.sub(r'[0-9]', '', sent)
                prev_snt = word_tokenize(re.sub(r'[^\w\s]', '', prev_snt))
                # prev_snt = [
                #     word for word in prev_snt if word not in stopwords.words('english')]
                prev_snt = [
                    word for word in prev_snt if word in self.embeddings_dict.keys()]
                if len(prev_snt) == 0:
                    prev_snt_embeds.append(np.zeros(self.emb_dim))
                else:
                    snt_embeds = list()
                    for word in prev_snt:
                        if word in self.embeddings_dict.keys():
                            snt_embeds.append(self.embeddings_dict[word])
                        else:
                            snt_embeds.append(np.zeros(self.emb_dim))
                    snt_embeds = np.array(snt_embeds)
                    prev_snt_embeds.append(snt_embeds.mean(axis=0))
        prev_snt_embeds = np.array(prev_snt_embeds)
        prev_snt_embeds = prev_snt_embeds.mean(axis=0)

        new_snt = re.sub(r'[0-9]', '', new_snt)
        new_snt = word_tokenize(re.sub(r'[^\w\s]', '', new_snt))
        # new_snt = [
        #     word for word in new_snt if word not in stopwords.words('english')]
        new_snt = [
            word for word in new_snt if word in self.embeddings_dict.keys()]
        if len(new_snt) == 0:
            new_snt_embeds = np.zeros(self.emb_dim)
        else:
            new_snt_embeds = []
            for word in new_snt:
                if word in self.embeddings_dict.keys():
                    new_snt_embeds.append(self.embeddings_dict[word])
                else:
                    new_snt_embeds.append(np.zeros(25))
            new_snt_embeds = np.array(new_snt_embeds)
            new_snt_embeds = new_snt_embeds.mean(axis=0)

        return np.concatenate((prev_snt_embeds, new_snt_embeds))

    def form_embeddings_dict(self):
        embeddings_dict = {}
        with open("glove.twitter.27b.25d.txt", 'r') as f:
            for line in f:
                values = line.split()
                word = values[0]
                vector = np.asarray(values[1:], "float32")
                embeddings_dict[word] = vector
        return embeddings_dict

    def get_imbalance_ratio(self):
        minority_count = sum(self.train_labels)
        majority_count = len(self.train_labels) - minority_count
        return minority_count / majority_count

    def train_and_eval_LR(self):
        log_reg = LogisticRegression()
        log_reg.fit(self.train_x, self.train_labels)
        predicted_y = log_reg.predict_proba(self.valid_x)
        return predicted_y

    def train_and_eval_RF(self):
        rf = RandomForestClassifier()
        rf.fit(self.train_x, self.train_labels)
        return rf.predict_proba(self.valid_x)

    def train_and_evalNB(self):
        nb = GaussianNB()
        nb.fit(self.train_x, self.train_labels)
        return nb.predict_proba(self.valid_x)

    def train_and_eval_SVC(self):
        svc = SVC(probability=True, class_weight='balanced')
        svc.fit(self.train_x, self.train_labels)
        return svc.predict_proba(self.valid_x)

    def fit_on_entire_training_set(self):
        self.LR = LogisticRegression()
        all_x = list()
        all_x += self.train_x
        all_x += self.valid_x
        all_labels = list()
        all_labels += self.train_labels
        all_labels += self.valid_labels
        self.LR.fit(all_x, all_labels)
        print("model fitted on all data")

    def evaluate(self):
        test_x, test_labels = self.form_test_data()
        pred_probs = self.LR.predict_proba(test_x)
        pred_ad_prob = [prob[-1] for prob in pred_probs]
        predicted_class = [0 if prob < 0.5 else 1 for prob in pred_ad_prob]

        i = 0
        tp = 0
        fp = 0
        fn = 0
        tp_list = list()
        fp_list = list()
        fn_list = list()
        for correct_label, probs in zip(test_labels, pred_probs):
            i += 1
            content_prob = probs[0]
            ad_prob = probs[1]
            if ad_prob > content_prob and correct_label == 1:
                tp += 1
                tp_list.append(ad_prob)
            if ad_prob < content_prob and correct_label == 1:
                fn += 1
                fn_list.append(ad_prob)
            if ad_prob > content_prob and correct_label == 0:
                fp += 1
                fp_list.append(ad_prob)
        overall_roc_auc = roc_auc_score(test_labels, pred_ad_prob)
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)

        print(
            f"AUROC score for all the predictions: {overall_roc_auc}")
        print(f"precision: {precision}, recall: {recall}")
        print(
            f"average confidence for correctly labeled ads (TP): {sum(tp_list) / len(tp_list)}")
        print(
            f"average confidence for incorrectly labeled content 'O' (FN): {sum(fn_list) / len(fn_list)}")
        print(
            f"average confidence for incorrectely labeled ads (FP): {sum(fp_list) / len(fp_list)}")


if __name__ == "__main__":
    model = AdDetection()
    # pred_probs = model.train_and_eval_LR()
    # pred_probs = model.train_and_eval_RF()
    # pred_probs = model.train_and_eval_NB()
    # pred_probs = model.train_and_eval_SVC()
    # y = zip(ad_detect.valid_labels, pred_y)
    # print(pred_y)
    pred_ad_prob = [prob[-1] for prob in pred_probs]
    predicted_class = [0 if prob < 0.5 else 1 for prob in pred_ad_prob]
    i = 0
    tp = 0
    fp = 0
    fn = 0
    tp_list = list()
    fp_list = list()
    fn_list = list()
    for correct_label, probs in zip(model.valid_labels, pred_probs):
        i += 1
        content_prob = probs[0]
        ad_prob = probs[1]
        if ad_prob > content_prob and correct_label == 1:
            tp += 1
            tp_list.append(ad_prob)
        if ad_prob < content_prob and correct_label == 1:
            fn += 1
            fn_list.append(ad_prob)
        if ad_prob > content_prob and correct_label == 0:
            fp += 1
            fp_list.append(ad_prob)

    overall_roc_auc = roc_auc_score(model.valid_labels, pred_ad_prob)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    print(
        f"AUROC score for all the predictions: {overall_roc_auc}")
    print(f"precision: {precision}, recall: {recall}")
    print(
        f"average confidence for correctly labeled ads (TP): {sum(tp_list) / len(tp_list)}")
    print(
        f"average confidence for incorrectly labeled content 'O' (FN): {sum(fn_list) / len(fn_list)}")
    print(
        f"average confidence for incorrectely labeled ads (FP): {sum(fp_list) / len(fp_list)}")
