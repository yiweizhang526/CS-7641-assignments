import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt


from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
from sklearn import model_selection


class MLPAnalysis:
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
	
		mlp_model = MLPClassifier(random_state=self.seed)
		mlp_model.fit(self.train_X, self.train_y)
		pred = mlp_model.predict(self.test_X)
		print("Recall: " + str(metrics.recall_score(self.test_y, pred)))
	
	def GridSearch_hidden_layer_sizes(self):
		'''
		searching for suitable hidden_layer_sizes
		'''
		mlp_model1 = MLPClassifier(batch_size=50, random_state=self.seed, max_iter=1000)
		nodes = [(20), (40), (60), (80), (100), (120), (140), (160),
				 (10, 10), (20, 20), (30, 30), (40, 40), (50, 50), (60, 60), (70, 70), (80, 80), (90, 90), (100, 100),
				 (30, 10), (50, 30), (50, 10), (70, 50), (70, 30), (70, 10), (90, 70), (90, 50), (90, 30), (90, 10)]

		tuned_parameters = {'hidden_layer_sizes': nodes}
		clf = model_selection.GridSearchCV(mlp_model1, tuned_parameters, scoring="recall", n_jobs=self.n_jobs,
										   cv=self.cv)
		clf.fit(self.train_X, self.train_y)
		print(clf.best_score_, clf.best_params_)
	
	def check_gridsearch_learning_curve(self):
		mlp_model2 = MLPClassifier(hidden_layer_sizes=(70, 30), batch_size=50, random_state=self.seed, max_iter=1000)
		
		train_sizes = np.linspace(0.1, 1.0, 10)
		train_sizes_abs, train_scores, valid_scores = model_selection.learning_curve(mlp_model2,
																					 self.train_X, self.train_y,
																					 cv=self.cv, n_jobs=self.n_jobs,
																					 train_sizes=train_sizes,
																					 random_state=self.seed,
																					 scoring="recall")
		
		plt.figure()
		plt.plot(train_sizes_abs, train_scores.mean(axis=1), 'o-', color='r', label="training score")
		plt.plot(train_sizes_abs, valid_scores.mean(axis=1), 'o-', color='g', label="cross-validation score")
		plt.xlabel('Size of training set')
		plt.ylabel('Recall score')
		plt.legend(loc="best")
		plt.title("Multiple layer perceptron learning curve (sizes=(70, 30))")
		plt.grid()
		plt.show()
	
	def learning_rate_validation(self):
		'''
		searching for suitable learning rate
		'''
		mlp_model = MLPClassifier(hidden_layer_sizes=(70, 30), batch_size=50, random_state=self.seed, max_iter=1000)
		
		lr = np.arange(0.0005, 0.01, 0.0005)
		train_scores, valid_scores = model_selection.validation_curve \
			(mlp_model, self.train_X, self.train_y, param_name="learning_rate_init", param_range=lr, cv=self.cv,
			 n_jobs=self.n_jobs, scoring="recall")
		
		plt.figure()
		plt.plot(lr, train_scores.mean(axis=1), 'r', label="training score")
		plt.plot(lr, valid_scores.mean(axis=1), 'g', label="cross-validation score")
		plt.xlabel('learning rate')
		plt.ylabel('Recall Score')
		plt.legend(loc="best")
		plt.title("Model Complexity: selecting learning rate (sizes=(70, 30))")
		plt.grid()
		plt.show()
	
	def GridSearch_with_activation(self):
		'''
		learning_rate_init, activation
		'''
		mlp_model = MLPClassifier(hidden_layer_sizes=(70, 30), batch_size=50, random_state=self.seed, max_iter=1000)
		tuned_parameters = {'learning_rate_init': np.arange(0.001, 0.004, 0.0005),
							'activation': ["relu", "tanh", "logistic"]}
		clf = model_selection.GridSearchCV(mlp_model, tuned_parameters, scoring="recall", n_jobs=self.n_jobs,
										   cv=self.cv)
		clf.fit(self.train_X, self.train_y)
		print(clf.best_score_, clf.best_params_)
	
	def learning_curve(self):
		'''
		training/testing error in two dimension (train sizes, iteration)
		'''
		mlp_model = MLPClassifier(activation="relu", learning_rate_init=0.001, hidden_layer_sizes=(70, 30),
								  batch_size=50, random_state=self.seed, max_iter=1000)
		# mlp_model.fit(self.train_X, self.train_y)
		
		train_sizes = np.linspace(0.1, 1.0, 10)
		train_sizes_abs, train_scores, valid_scores = model_selection.learning_curve(mlp_model,
																					 self.train_X, self.train_y,
																					 cv=self.cv, n_jobs=self.n_jobs,
																					 train_sizes=train_sizes,
																					 random_state=self.seed,
																					 scoring="recall")
		max_iter = np.arange(20, 500, 20)
		train_scores2, valid_scores2 = model_selection.validation_curve \
			(mlp_model, self.train_X, self.train_y, param_name="max_iter", param_range=max_iter, cv=self.cv,
			 n_jobs=self.n_jobs, scoring="recall")
		
		plt.figure(figsize=(8, 10))
		plt.subplot(211)
		plt.plot(train_sizes_abs, train_scores.mean(axis=1), 'o-', color='r', label="training score")
		plt.plot(train_sizes_abs, valid_scores.mean(axis=1), 'o-', color='g', label="cross-validation score")
		plt.xlabel('Size of training set')
		plt.ylabel('Recall score')
		plt.legend(loc="best")
		plt.title("Multiple layer perceptron learning curve 1 (sizes=(70, 30))")
		plt.grid()
		plt.subplot(212)
		plt.plot(max_iter, train_scores2.mean(axis=1), 'o-', color='r', label="training score")
		plt.plot(max_iter, valid_scores2.mean(axis=1), 'o-', color='g', label="cross-validation score")
		plt.xlabel('Iterations ("max_iter")')
		plt.ylabel('Recall score')
		plt.legend(loc="best")
		plt.title("Multiple layer perceptron learning curve 2 (sizes=(70, 30))")
		plt.grid()
		plt.show()
	
	def early_stopping(self):
		mlp_model1 = MLPClassifier(activation="relu", learning_rate_init=0.001, hidden_layer_sizes=(70, 30),
								  batch_size=50, random_state=self.seed, max_iter=1000)
		mlp_model1.fit(self.train_X, self.train_y)
		pred = mlp_model1.predict(self.test_X)
		print("Loss: " + str(mlp_model1.loss_) + "(" + str(mlp_model1.best_loss_) + ")",
			  "Recall: " + str(metrics.recall_score(self.test_y, pred)))
		mlp_model2 = MLPClassifier(activation="relu", learning_rate_init=0.001, hidden_layer_sizes=(70, 30),
								   batch_size=50, random_state=self.seed, max_iter=1000, early_stopping=True, n_iter_no_change=50)
		mlp_model2.fit(self.train_X, self.train_y)
		pred = mlp_model2.predict(self.test_X)
		print("Loss: " + str(mlp_model2.loss_) + "(" + ")",
			  "Recall: " + str(metrics.recall_score(self.test_y, pred)))
		
	def compare_and_output(self):
		'''
		selecting the best model, recording run time
		recording confusion matrix, calculate Precision, Recall, F1 score
		plot P-R, ROC
		'''
		mlp_model = MLPClassifier(activation="relu", learning_rate_init=0.001, hidden_layer_sizes=(70, 30),
								  batch_size=50, random_state=self.seed, max_iter=1000)
		t0_clock = time.process_time()
		mlp_model.fit(self.train_X, self.train_y)
		pred = mlp_model.predict(self.test_X)  # Predict with test set
		t1_clock = time.process_time()
		print("The training time for final selected model is " + str(t1_clock - t0_clock) + " seconds")
		print(metrics.classification_report(self.test_y, pred, digits=4))



if __name__ == '__main__':
	data = pd.read_csv("../winequality.csv")
	label = "label"
	MLP = MLPAnalysis(data, label)
	
	MLP.splitting_data()
	MLP.GridSearch_hidden_layer_sizes()
	MLP.check_gridsearch_learning_curve()
	MLP.learning_rate_validation()
	MLP.GridSearch_with_activation()
	MLP.learning_curve()
	MLP.compare_and_output()
