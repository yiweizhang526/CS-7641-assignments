import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
from sklearn import model_selection


class SVMAnalysis:
	def __init__(self, data, label):
		self.data = data
		self.label = label
		self.seed = 77
		self.cv = 10
		self.n_jobs = -1
		
		self.train_X, self.train_y = [], []
		self.test_X, self.test_y = [], []
	
	def splitting_data(self):
		predictors = self.data[self.data.columns.difference([self.label])]
		scaler = MinMaxScaler()
		predictors_scaled = scaler.fit_transform(predictors)
		target = self.data[[self.label]].values.ravel()
		
		self.train_X, self.test_X, self.train_y, self.test_y = model_selection.train_test_split \
			(predictors_scaled, target, train_size=0.8, random_state=self.seed, stratify=target)
		
		svm_model = SVC(random_state=self.seed)
		svm_model.fit(self.train_X, self.train_y)
		pred = svm_model.predict(self.test_X)
		print("Recall: " + str(metrics.recall_score(self.test_y, pred)))
	
	def C_validation(self):
		'''
		searching for suitable C
		'''
		svm_model = SVC(kernel='linear', random_state=self.seed)
		C = np.arange(1, 50, 1)
		train_scores, valid_scores = model_selection.validation_curve \
			(svm_model, self.train_X, self.train_y, param_name="C", param_range=C, cv=self.cv,
			 n_jobs=self.n_jobs, scoring="recall")
		
		plt.figure()
		plt.plot(C, train_scores.mean(axis=1), color='r', label="training score")
		plt.plot(C, valid_scores.mean(axis=1), color='g', label="cross-validation score")
		plt.xlabel('C')
		plt.ylabel('Recall Score')
		plt.legend(loc="best")
		plt.title("Model Complexity: selecting C (kernel='linear')")
		plt.grid()
		plt.show()
	
	def Grid_Search_with_kernel(self):
		'''
		C, gamma
		'''
		svm_model = SVC(kernel='linear', random_state=self.seed)
		tuned_parameters = {'C': range(20, 30), 'gamma': ["scale", "auto"]}
		# tuned_parameters = {'C': range(1, 6), 'gamma': ["scale", "auto"]}
		clf = model_selection.GridSearchCV(svm_model, tuned_parameters, scoring="recall", n_jobs=self.n_jobs,
						   cv=self.cv)
		clf.fit(self.train_X, self.train_y)
		print(clf.best_score_, clf.best_params_)
	
	def learning_curve(self):
		'''
		training/testing error in two dimension (train sizes, iteration)
		'''
		# set verbose = True
		svm_model1 = SVC(C=2, kernel="rbf", gamma="scale", random_state=self.seed)
		svm_model2 = SVC(C=24, kernel="linear", gamma="scale", random_state=self.seed, verbose=True)

		# svm_model2.fit(self.train_X, self.train_y)
		
		train_sizes = np.linspace(0.1, 1.0, 10)
		train_sizes_abs, train_scores1, valid_scores1 = model_selection.learning_curve(svm_model2,
																					self.train_X, self.train_y,
																					cv=self.cv, n_jobs=self.n_jobs,
																					train_sizes=train_sizes,
																					random_state=self.seed,
																					scoring="recall")
		
		max_iter = np.arange(0, 12000, 400)
		train_scores2, valid_scores2 = model_selection.validation_curve \
			(svm_model2, self.train_X, self.train_y, param_name="max_iter", param_range=max_iter, cv=self.cv,
			 n_jobs=self.n_jobs, scoring="recall")
		
		plt.figure(figsize=(8, 10))
		plt.subplot(211)
		plt.plot(train_sizes_abs, train_scores1.mean(axis=1), 'o-', color='r', label="training score")
		plt.plot(train_sizes_abs, valid_scores1.mean(axis=1), 'o-', color='g', label="cross-validation score")
		plt.xlabel('Size of training set')
		plt.ylabel('Recall score')
		plt.legend(loc="best")
		plt.title("Support Vector Machine learning curve 1 (kernel='linear')")
		plt.grid()
		plt.subplot(212)
		plt.plot(max_iter, train_scores2.mean(axis=1), 'o-', color='r', label="training score")
		plt.plot(max_iter, valid_scores2.mean(axis=1), 'o-', color='g', label="cross-validation score")
		plt.xlabel('Iterations ("max_iter")')
		plt.ylabel('Recall score')
		plt.legend(loc="best")
		plt.title("Support Vector Machine learning curve 2 (kernel='linear')")
		plt.grid()
		plt.show()
	
	def compare_and_output(self):
		'''
		test the best model
		'''
		svm_model = SVC(C=2, kernel="rbf", gamma="scale", random_state=self.seed)
		t0_clock = time.process_time()
		svm_model.fit(self.train_X, self.train_y)
		pred = svm_model.predict(self.test_X)  # Predict with test set
		t1_clock = time.process_time()
		print("The training time for final selected model is " + str(t1_clock - t0_clock) + " seconds")
		print(metrics.classification_report(self.test_y, pred, digits=4))


if __name__ == '__main__':
	data = pd.read_csv("../winequality.csv")
	label = "label"
	SVM = SVMAnalysis(data, label)
	
	SVM.splitting_data()
	SVM.C_validation()
	SVM.Grid_Search_with_kernel()
	SVM.learning_curve()
	SVM.compare_and_output()
