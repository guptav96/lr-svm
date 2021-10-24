"""Naive Bayes Classifier Implementation"""
import pandas as pd
import numpy as np

from lr_svm import accuracy

laplace_correction = True

def nbc(train_df, test_df):
    attributes = train_df.columns[:-1]

    # calculating prior probability
    prior = train_df.groupby(by = 'decision').size().div(len(train_df))

    # calculating conditional probabilities
    conditional_prob = {}
    for attribute in attributes:
        numerator = train_df.groupby(by = ['decision'])[attribute].value_counts().unstack('decision')
        denominator = numerator.sum()
        k = train_df[attribute].nunique()
        if laplace_correction and numerator.isna().any(axis=None):
            numerator.fillna(value=0, inplace=True)
            numerator += 1
            denominator += k
        conditional_prob[attribute] = numerator.div(denominator)

    # calculating posterior probability for all training examples
    def predict(row, label):
        result = 1
        for attribute in attributes:
            try:
                result *= conditional_prob[attribute][label][row[attribute]]
            except:
                # if there is a new attribute value not known at the training time
                # laplace correction wouldn't work here, since the attr val is not known at the training time
                continue
        result *= prior[label]
        return result

    predicted_training_no = np.array([ predict(row, 0) for idx, row in train_df.iterrows() ])
    predicted_training_yes =  np.array([ predict(row, 1) for idx, row in train_df.iterrows() ])
    predicted_training_labels = predicted_training_yes > predicted_training_no
    train_accuracy = accuracy(train_df[['decision']].to_numpy(), predicted_training_labels.reshape(-1, 1))

    predicted_test_no = np.array([ predict(row, 0) for idx, row in test_df.iterrows() ])
    predicted_test_yes =  np.array([ predict(row, 1) for idx, row in test_df.iterrows() ])
    predicted_test_labels = predicted_test_yes > predicted_test_no
    test_accuracy = accuracy(test_df[['decision']].to_numpy(), predicted_test_labels.reshape(-1, 1))

    return train_accuracy, test_accuracy
