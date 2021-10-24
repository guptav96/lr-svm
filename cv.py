import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt

from lr_svm import lr, svm
from nbc import nbc

def shuffle_train(train_df, frac, random_state = 18):
	train_df.sample(frac = frac, random_state = random_state)

	# partition training data into 10 disjoint sets
	sample_size = int(train_df.shape[0]/10)
	S = [train_df.iloc[i*sample_size:(i+1)*sample_size,:] for i in range(10)]

	return S

def run_models(model_idx, S, t_fracs):
	accuracy_frac_dict = {}
	std_error_frac_dict = {}
	for t_frac in t_fracs:
		accuracy_cv = []
		for idx in range(10):
			test_set = S[idx]
			list_ = []
			for item in range(10):
				not item == idx and list_.append(item)
			S_c = pd.concat(S[item] for item in list_)
			train_set = S_c.sample(frac = t_frac, random_state = 32)
			if model_idx == 1:
				_, test_acc = lr(train_set, test_set)
			elif model_idx == 2:
				_, test_acc = svm(train_set, test_set)
			else:
				_, test_acc = nbc(train_set, test_set)
			accuracy_cv.append(test_acc)
		avg_accuracy = sum(accuracy_cv)/len(accuracy_cv)
		variance = sum([((x - avg_accuracy) ** 2) for x in accuracy_cv]) / len(accuracy_cv)
		std_error = (variance ** 0.5) / math.sqrt(10)
		accuracy_frac_dict[t_frac*S_c.shape[0]] = avg_accuracy
		std_error_frac_dict[t_frac*S_c.shape[0]] = std_error
	return accuracy_frac_dict, std_error_frac_dict

def plot_curves(nbc_acc_dict, nbc_err_dict, lr_acc_dict, lr_err_dict, svm_acc_dict, svm_err_dict):
	plt.errorbar(nbc_acc_dict.keys(), nbc_acc_dict.values(), yerr = nbc_err_dict.values(), label='NBC')
	plt.errorbar(lr_acc_dict.keys(), lr_acc_dict.values(), yerr = lr_err_dict.values(), label='LR')
	plt.errorbar(svm_acc_dict.keys(), svm_acc_dict.values(), yerr = svm_err_dict.values(), label='SVM')
	plt.legend(loc='upper left')
	plt.xlabel('Size of the training set')
	plt.ylabel('Model Accuracy and Standard Error')
	plt.title('Learning Curves for each model: NBC, LR and SVM')
	plt.show()

if __name__ == '__main__':
	train_df = pd.read_csv('trainingSet.csv')
	train_df_nbc = pd.read_csv('trainingSet_NBC.csv')
	# part (i)
	S = shuffle_train(train_df, 1, 18)
	S_nbc = shuffle_train(train_df_nbc, 1, 18)
    # part (ii)
	t_fracs = [0.025, 0.05, 0.075, 0.1, 0.15, 0.2]
	# model_idx = 0 for NBC, 1 for LR, 2 for SVM
	nbc_acc_dict, nbc_err_dict = run_models(0, S_nbc, t_fracs) # train and test NBC
	lr_acc_dict, lr_err_dict = run_models(1, S, t_fracs) # train and test LR
	svm_acc_dict, svm_err_dict = run_models(2, S, t_fracs) # train and test SVM
	# part(iii)
	plot_curves(nbc_acc_dict, nbc_err_dict, lr_acc_dict, lr_err_dict, svm_acc_dict, svm_err_dict)
	# part(iv)
