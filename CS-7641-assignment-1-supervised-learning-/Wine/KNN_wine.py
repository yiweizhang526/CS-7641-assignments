import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
from sklearn import model_selection


class KNNAnalysis:
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
			(predictors_scaled, target, train_size=0.7, random_state=self.seed, stratify=target)
		
		neighbor_model = KNeighborsClassifier()
		neighbor_model.fit(self.train_X, self.train_y)
		pred = neighbor_model.predict(self.test_X)
		print("Recall: " + str(metrics.recall_score(self.test_y, pred)))
	
	def k_validation(self):
		'''
		searching for suitable k
		'''
		neighbor_model = KNeighborsClassifier(weights="distance")
		n_neighbors = np.arange(1, 50, 1)
		train_scores, valid_scores = model_selection.validation_curve \
			(neighbor_model, self.train_X, self.train_y, param_name="n_neighbors", param_range=n_neighbors, cv=self.cv,
			 n_jobs=self.n_jobs, scoring="recall")
		
		plt.figure()
		plt.plot(n_neighbors, train_scores.mean(axis=1), 'r', label="training score")
		plt.plot(n_neighbors, valid_scores.mean(axis=1), 'g', label="cross-validation score")
		plt.xlabel('n_neighbors')
		plt.ylabel('Recall score')
		plt.legend(loc="best")
		plt.title("Model Complexity: selecting K (weights='distance')")
		plt.grid()
		plt.show()
	
	def GridSearch_with_metric(self):
		'''
		n_neighbors, metric
		'''
		neighbor_model = KNeighborsClassifier(weights="distance")
		tuned_parameters = {'n_neighbors': range(20, 30), 'metric': ["manhattan", "chebyshev", "euclidean"]}
		clf = model_selection.GridSearchCV(neighbor_model, tuned_parameters, scoring="recall", n_jobs=self.n_jobs,
						   cv=self.cv)
		clf.fit(self.train_X, self.train_y)
		print(clf.best_score_, clf.best_params_)
	
	def learning_curve_and_output(self):
		'''
		test the best model
		'''
		neighbor_model = KNeighborsClassifier(n_neighbors=28, weights="distance", metric="manhattan")
		
		train_sizes = np.linspace(0.1, 1.0, 10)
		train_sizes_abs, train_scores, valid_scores = model_selection.learning_curve(neighbor_model,
																					 self.train_X, self.train_y,
																					 cv=self.cv, n_jobs=self.n_jobs,
																					 train_sizes=train_sizes,
																					 random_state=self.seed,
																					 scoring="recall")
		
		plt.figure()
		plt.plot(train_sizes_abs, train_scores.mean(axis=1), 'o-', color='r', label="training score")
		plt.plot(train_sizes_abs, valid_scores.mean(axis=1), 'o-', color='g', label="cross-validation score")
		plt.xlabel('Size of training set')
		plt.ylabel('Recall Score')
		plt.legend(loc="best")
		plt.title("K Nearest Neighbor learning curve")
		plt.grid()
		plt.show()
		
		
		# classification report
		t0_clock = time.process_time()
		neighbor_model.fit(self.train_X, self.train_y)
		pred = neighbor_model.predict(self.test_X)  # Predict with test set
		t1_clock = time.process_time()
		print("The training time for final selected model is " + str(t1_clock - t0_clock) + " seconds")
		print(metrics.classification_report(self.test_y, pred, digits=4))


if __name__ == '__main__':
	data = pd.read_csv("../winequality.csv")
	label = "label"
	KNN = KNNAnalysis(data, label)
	
	KNN.splitting_data()
	KNN.k_validation()
	KNN.GridSearch_with_metric()
	KNN.learning_curve_and_output()
