import sys
import warnings

#suppress warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
# np.set_printoptions(precision=16)

lr_options = {
	'lambda': 0.01,
	'step_size': 0.01,
	'max_iter': 500,
	'tol': 1e-6,
}

svm_options = {
	'lambda': 0.01,
	'step_size': 0.5,
	'max_iter': 500,
	'tol': 1e-6
}

def logistic(x):
	# default is also float64
	return 1./(1+np.exp(-x)).astype(np.float64)

def l2_norm(x):
	return np.linalg.norm(x.ravel(), ord=2)

def accuracy(original_labels, predicted_labels):
    count = 0
    total_num = len(original_labels)
    for idx in range(total_num):
        if original_labels[idx] == predicted_labels[idx]:
            count += 1
    return float(count)/total_num

def process(train_df, test_df):
	# get data attribute values and labels
	train_X = train_df.drop(columns='decision').to_numpy(copy = False)
	train_y = train_df[['decision']].to_numpy(copy = False)

	test_X = test_df.drop(columns='decision').to_numpy(copy = False)
	test_y = test_df[['decision']].to_numpy(copy = False)

	# insert an intercept column for bias
	train_X = np.insert(train_X, 0, 1, axis = 1)
	test_X = np.insert(test_X, 0, 1, axis = 1)

	return train_X, train_y, test_X, test_y

def lr(train_df, test_df):
	train_X, train_y, test_X, test_y = process(train_df, test_df)
	# initialize weights to zeros
	weights = np.zeros((train_X.shape[1], 1))

	# training by gradient descent
	for _ in range(lr_options['max_iter']):
		pred_y = logistic(np.dot(train_X, weights))
		grad_ = np.dot(train_X.T , (pred_y - train_y)) + lr_options['lambda'] * weights

		# check if diff in weight norm is more than tol
		delta_w = lr_options['step_size'] *  grad_
		if l2_norm(delta_w) < lr_options['tol']:
			break
		weights = weights - delta_w

	# testing
	pred_test_y = logistic(np.dot(test_X, weights))
	pred_test_y = np.where(pred_test_y > 0.5, 1, 0)

	pred_train_y = logistic(np.dot(train_X, weights))
	pred_train_y = np.where(pred_train_y > 0.5, 1, 0)

	# getting accuracy of model
	train_accuracy = accuracy(pred_train_y, train_y)
	test_accuracy = accuracy(pred_test_y, test_y)

	return train_accuracy, test_accuracy

def svm(train_df, test_df):
	train_X, train_y, test_X, test_y = process(train_df, test_df)
	# change labels to -1 and 1 for SVM
	train_y = np.where(train_y == 0, -1, 1)
	N = train_X.shape[0]

	# initialize weights to zeros
	weights = np.zeros((train_X.shape[1], 1))

	# training by sub-gradient descent
	for _ in range(svm_options['max_iter']):
		pred_y = np.dot(train_X, weights)
		mask = train_y * pred_y
		delta = np.where(mask < 1, train_y * train_X, 0)
		grad_ = 1./N * (svm_options['lambda'] * N * weights - np.sum(delta, axis = 0).reshape(-1,1))
		# check if diff in weight norm is more than tol
		delta_w = svm_options['step_size'] * grad_
		if l2_norm(delta_w) < svm_options['tol']:
			break
		weights = weights - delta_w

	# testing
	pred_test_y = np.dot(test_X, weights)
	pred_test_y = np.where(pred_test_y > 0, 1, 0)

	# change labels back to original labels
	train_y = np.where(train_y == -1, 0, 1)
	pred_train_y = np.dot(train_X, weights)
	pred_train_y = np.where(pred_train_y > 0, 1, 0)

	# print(np.unique(pred_train_y, return_counts = True))
	# print(np.unique(pred_test_y, return_counts = True))

	# getting accuracy of model
	train_accuracy = accuracy(pred_train_y, train_y)
	test_accuracy = accuracy(pred_test_y, test_y)

	return train_accuracy, test_accuracy

def perform(train_df, test_df, model_idx):
	if model_idx == 1:
		train_acc, test_acc = lr(train_df, test_df)
		print(f'Training Accuracy LR: {round(train_acc,2)}')
		print(f'Testing Accuracy LR: {round(test_acc,2)}')
	else:
		train_acc, test_acc = svm(train_df, test_df)
		print(f'Training Accuracy SVM: {round(train_acc,2)}')
		print(f'Testing Accuracy SVM: {round(test_acc,2)}')

if __name__ == '__main__':
	training_data_filename, test_data_filename, model_idx = sys.argv[1], sys.argv[2], sys.argv[3]
	train_df = pd.read_csv(training_data_filename)
	test_df = pd.read_csv(test_data_filename)
	# model_idx = 1 for Logistic Regression, 2 for SVM classification
	perform(train_df, test_df, int(model_idx))
