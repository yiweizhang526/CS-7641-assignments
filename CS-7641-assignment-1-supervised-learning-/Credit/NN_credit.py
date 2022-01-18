import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt

from pandas.api.types import is_string_dtype
from pandas.api.types import is_categorical_dtype

from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.exceptions import ConvergenceWarning
from sklearn import metrics
from sklearn import model_selection


class MLPAnalysis:
	def __init__(self, data, label):
		self.data = data
		self.label = label
		self.seed = 166
		self.cv = 5
		self.n_jobs = -1
		
		self.train_X, self.train_y = [], []
		self.test_X, self.test_y = [], []
	
	def splitting_data(self):
		predictors = self.data[self.data.columns.difference([self.label])]
		scaler = MinMaxScaler()
		predictors_scaled = scaler.fit_transform(predictors)
		target = self.data[[self.label]].values.ravel()
		
		self.train_X, self.test_X, self.train_y, self.test_y = model_selection.train_test_split \
			(predictors_scaled, target, train_size=0.9, random_state=self.seed, stratify=target)
	
		mlp_model = MLPClassifier(random_state=self.seed, max_iter=800)
		mlp_model.fit(self.train_X, self.train_y)
		pred = mlp_model.predict(self.test_X)
		print("f1: " + str(metrics.f1_score(self.test_y, pred)))
	
	def GridSearch_hidden_layer_sizes(self):
		'''
		searching for suitable hidden_layer_sizes
		'''
		mlp_model1 = MLPClassifier(random_state=self.seed, max_iter=1000)
		nodes = [(40, 40), (50, 50), (60, 60), (70, 70), (80, 80), (90, 90), (100, 100), (120, 120), (150, 150),
				 (80, 10), (80, 30), (80, 50), (100, 10), (100, 30), (100, 50), (120, 70), (120, 50), (150, 50),
				 (150, 100), (150, 70)]
		batch_size = [50, 100]

		tuned_parameters = {'hidden_layer_sizes': nodes, 'batch_size': batch_size}
		clf = model_selection.GridSearchCV(mlp_model1, tuned_parameters, scoring="f1", n_jobs=self.n_jobs,
										   cv=self.cv)
		clf.fit(self.train_X, self.train_y)
		print(clf.best_score_, clf.best_params_)
		
	def check_gridsearch_learning_curve(self):
		mlp_model2 = MLPClassifier(hidden_layer_sizes=(60, 60), batch_size=50, random_state=self.seed, max_iter=1000)
	
		train_sizes = np.linspace(0.1, 1.0, 10)
		train_sizes_abs, train_scores, valid_scores = model_selection.learning_curve(mlp_model2,
																					 self.train_X, self.train_y,
																					 cv=self.cv, n_jobs=self.n_jobs,
																					 train_sizes=train_sizes,
																					 random_state=self.seed,
																					 scoring="f1")
	
		plt.figure()
		plt.plot(train_sizes_abs, train_scores.mean(axis=1), 'o-', color='r', label="training score")
		plt.plot(train_sizes_abs, valid_scores.mean(axis=1), 'o-', color='g', label="cross-validation score")
		plt.xlabel('Size of training set')
		plt.ylabel('f1 score')
		plt.legend(loc="best")
		plt.title("Multiple layer perceptron learning curve (sizes=(60, 60))")
		plt.grid()
		plt.show()
	
	def learning_rate_test(self):
		'''
		searching for suitable learning rate
		'''
		mlp_model = MLPClassifier(hidden_layer_sizes=(60, 60), batch_size=50, random_state=self.seed, max_iter=1000)
		
		lr = np.arange(0.0005, 0.01, 0.0005)
		train_scores, valid_scores = model_selection.validation_curve \
			(mlp_model, self.train_X, self.train_y, param_name="learning_rate_init", param_range=lr, cv=self.cv,
			 n_jobs=self.n_jobs, scoring="f1")
		
		plt.figure()
		plt.plot(lr, train_scores.mean(axis=1), 'r', label="training score")
		plt.plot(lr, valid_scores.mean(axis=1), 'g', label="cross-validation score")
		plt.xlabel('learning rate')
		plt.ylabel('f1 Score')
		plt.legend(loc="best")
		plt.title("Multiple layer perceptron: selecting learning rate (sizes=(60, 60))")
		plt.grid()
		plt.show()
	
	def GridSearch_with_activation(self):
		'''
		learning_rate_init, activation
		'''
		mlp_model = MLPClassifier(hidden_layer_sizes=(60, 60), batch_size=50, random_state=self.seed, max_iter=1000)
		tuned_parameters = {'learning_rate_init': np.arange(0.001, 0.008, 0.0005),
							'activation': ["relu", "tanh", "logistic"]}
		clf = model_selection.GridSearchCV(mlp_model, tuned_parameters, scoring="f1", n_jobs=self.n_jobs,
										   cv=self.cv)
		clf.fit(self.train_X, self.train_y)
		print(clf.best_score_, clf.best_params_)
	
	def learning_curve(self):
		'''
		training/testing error in two dimension (train sizes, iteration)
		'''
		mlp_model = MLPClassifier(activation="relu", learning_rate_init=0.001, hidden_layer_sizes=(60, 60),
								  batch_size=50, random_state=self.seed, max_iter=1000)
		# mlp_model.fit(self.train_X, self.train_y) # set verbose=True
		train_sizes = np.linspace(0.1, 1.0, 10)
		train_sizes_abs, train_scores, valid_scores = model_selection.learning_curve(mlp_model,
																					 self.train_X, self.train_y,
																					 cv=self.cv, n_jobs=self.n_jobs,
																					 train_sizes=train_sizes,
																					 random_state=self.seed,
																					 scoring="f1")
		max_iter = np.arange(10, 300, 10)
		train_scores2, valid_scores2 = model_selection.validation_curve \
			(mlp_model, self.train_X, self.train_y, param_name="max_iter", param_range=max_iter, cv=self.cv,
			 n_jobs=self.n_jobs, scoring="f1")
		
		plt.figure(figsize=(8, 10))
		plt.subplot(211)
		plt.plot(train_sizes_abs, train_scores.mean(axis=1), 'o-', color='r', label="training score")
		plt.plot(train_sizes_abs, valid_scores.mean(axis=1), 'o-', color='g', label="cross-validation score")
		plt.xlabel('Size of training set')
		plt.ylabel('f1 score')
		plt.legend(loc="best")
		plt.title("Multiple layer perceptron learning curve 1 (sizes=(60, 60))")
		plt.grid()
		plt.subplot(212)
		plt.plot(max_iter, train_scores2.mean(axis=1), 'o-', color='r', label="training score")
		plt.plot(max_iter, valid_scores2.mean(axis=1), 'o-', color='g', label="cross-validation score")
		plt.xlabel('Iterations ("max_iter")')
		plt.ylabel('f1 score')
		plt.legend(loc="best")
		plt.title("Multiple layer perceptron learning curve 2 (sizes=(60, 60))")
		plt.grid()
		plt.show()
	
	def early_stopping(self):
		mlp_model1 = MLPClassifier(activation="relu", learning_rate_init=0.001, hidden_layer_sizes=(70, 30),
								   batch_size=50, random_state=self.seed, max_iter=1000)
		mlp_model1.fit(self.train_X, self.train_y)
		pred = mlp_model1.predict(self.test_X)
		print("Loss: " + str(mlp_model1.loss_) + "(" + str(mlp_model1.best_loss_) + ")",
			  "f1: " + str(metrics.f1_score(self.test_y, pred)))
		mlp_model2 = MLPClassifier(activation="relu", learning_rate_init=0.001, hidden_layer_sizes=(70, 30),
								   batch_size=50, random_state=self.seed, max_iter=1000, early_stopping=True,
								   n_iter_no_change=50)
		mlp_model2.fit(self.train_X, self.train_y)
		pred = mlp_model2.predict(self.test_X)
		print("Loss: " + str(mlp_model2.loss_) + "(" + ")",
			  "f1: " + str(metrics.f1_score(self.test_y, pred)))
	
	def compare_and_output(self):
		'''
		test the best model
		'''
		mlp_model = MLPClassifier(activation="relu", learning_rate_init=0.001, hidden_layer_sizes=(60, 60),
								  batch_size=50, random_state=self.seed, max_iter=1000)
		t0_clock = time.process_time()
		mlp_model.fit(self.train_X, self.train_y)
		pred = mlp_model.predict(self.test_X)  # Predict with test set
		t1_clock = time.process_time()
		print("The training time for final selected model is " + str(t1_clock - t0_clock) + " seconds")
		print(metrics.classification_report(self.test_y, pred, digits=4))


def pre_processing(credit):
	# preprocessing - convert data type and dummy coding
	cols = credit.columns
	isCat_Index = list()
	for col in cols:
		if is_string_dtype(credit[col]):
			credit[col] = credit[col].astype('category')
		isCat_Index.append(is_categorical_dtype(credit[col]))
	
	credit_d = pd.get_dummies(credit, columns=list(cols[isCat_Index]))
	print(credit_d.shape)
	return credit_d


if __name__ == '__main__':
	data = pd.read_csv("../credit.csv")
	label = "default"
	data = pre_processing(data)
	MLP = MLPAnalysis(data, label)
	
	MLP.splitting_data()
	MLP.GridSearch_hidden_layer_sizes()
	MLP.check_gridsearch_learning_curve()
	MLP.learning_rate_test()
	MLP.GridSearch_with_activation()
	MLP.learning_curve()
	MLP.compare_and_output()
