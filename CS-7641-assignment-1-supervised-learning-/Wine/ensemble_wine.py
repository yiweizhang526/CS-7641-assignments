import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn import metrics
from sklearn import model_selection


class AdaBoostAnalysis:
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
		target = self.data[[self.label]]
		
		self.train_X, self.test_X, self.train_y, self.test_y = model_selection.train_test_split \
			(predictors, target, train_size=0.8, random_state=self.seed, stratify=target)
	
		boost_model = AdaBoostClassifier(random_state=self.seed)
		boost_model.fit(self.train_X, self.train_y)
		pred = boost_model.predict(self.test_X)
		print("Recall: " + str(metrics.recall_score(self.test_y, pred)))
	
	def estimator_pruning(self):
		'''
		prior pruning: searching for min_samples_split, min_smaples_split
		'''
		
		tree_model = DecisionTreeClassifier(max_depth=2, criterion="entropy", random_state=self.seed)
		min_samples_split = np.arange(2, 11, 1)
		train_scores1, valid_scores1 = model_selection.validation_curve \
			(tree_model, self.train_X, self.train_y, param_name="min_samples_split", param_range=min_samples_split,
			 cv=self.cv, n_jobs=self.n_jobs, scoring="recall")
		min_samples_leaf = np.arange(1, 6, 1)
		train_scores2, valid_scores2 = model_selection.validation_curve \
			(tree_model, self.train_X, self.train_y, param_name="min_samples_leaf", param_range=min_samples_leaf,
			 cv=self.cv, n_jobs=self.n_jobs, scoring="recall")
		
		plt.figure(figsize=(6,8))
		plt.subplot(211)
		plt.plot(min_samples_split, train_scores1.mean(axis=1), 'r', label="training score")
		plt.plot(min_samples_split, valid_scores1.mean(axis=1), 'g', label="cross-validation score")
		plt.xlabel('Min samples split')
		plt.ylabel('Recall Score')
		plt.legend(loc="best")
		plt.title("Decision tree prior_pruning experiments: min samples split")
		plt.grid()
		plt.subplot(212)
		plt.plot(min_samples_leaf, train_scores2.mean(axis=1), 'r', label="training score")
		plt.plot(min_samples_leaf, valid_scores2.mean(axis=1), 'g', label="cross-validation score")
		plt.xlabel('Min samples leaf')
		plt.ylabel('Recall Score')
		plt.legend(loc="best")
		plt.title("Decision tree prior_pruning experiments: min samples leaf")
		plt.grid()
		plt.show()
	
	def GridSearch_estimators(self):
		'''
		max_depth, criterion, min_samples_split, min_samples_leaf
		'''
		tree_model = DecisionTreeClassifier(random_state=self.seed)
		tuned_parameters = {'max_depth': range(1, 12),
							'criterion': ["gini", "entropy"],
							'min_samples_split': range(2, 11),
							'min_samples_leaf': range(1, 6)}
		clf = model_selection.GridSearchCV(tree_model, tuned_parameters, scoring="recall", n_jobs=self.n_jobs,
										   cv=self.cv)
		clf.fit(self.train_X, self.train_y)
		print(clf.best_score_, clf.best_params_)
	
	def n_estimators_validation(self):
		'''
		searching for suitable n_estimators
		'''
		tree_model = DecisionTreeClassifier(max_depth=2, criterion="entropy", min_samples_leaf=1, min_samples_split=2,
											random_state=self.seed)
		boost_model = AdaBoostClassifier(base_estimator=tree_model, random_state=self.seed, algorithm="SAMME")
		
		n_estimators = np.arange(1, 50, 1)
		train_scores, valid_scores = model_selection.validation_curve \
			(boost_model, self.train_X, self.train_y, param_name="n_estimators", param_range=n_estimators, cv=self.cv,
			 n_jobs=self.n_jobs, scoring="recall")
		
		plt.figure()
		plt.plot(n_estimators, train_scores.mean(axis=1), 'o-', color='r', label="training score")
		plt.plot(n_estimators, valid_scores.mean(axis=1), 'o-', color='g', label="cross-validation score")
		plt.xlabel('n estimators')
		plt.ylabel('Recall Score')
		plt.legend(loc="best")
		plt.title("Adaboost pruning experiments: n_estimators (Decision Tree)")
		plt.grid()
		plt.show()
	
	def GridSearch_with_algorithm(self):
		'''
		n_estimators, algorithm
		'''
		tree_model = DecisionTreeClassifier(max_depth=2, criterion="entropy", min_samples_leaf=1, min_samples_split=2,
											random_state=self.seed)
		boost_model = AdaBoostClassifier(base_estimator=tree_model, random_state=self.seed)
		
		tuned_parameters = {'n_estimators': range(1, 10), 'algorithm': ["SAMME", "SAMME.R"]}
		clf = model_selection.GridSearchCV(boost_model, tuned_parameters, scoring="recall", n_jobs=self.n_jobs,
										   cv=self.cv)
		clf.fit(self.train_X, self.train_y)
		print(clf.best_score_, clf.best_params_)
	
	def learning_curve_and_output(self):
		'''
		test the best model
		'''
		tree_model = DecisionTreeClassifier(max_depth=2, criterion="entropy", min_samples_leaf=1, min_samples_split=2,
											random_state=self.seed)
		boost_model = AdaBoostClassifier(base_estimator=tree_model, random_state=self.seed, n_estimators=1, algorithm="SAMME")
		
		train_sizes = np.linspace(0.1, 1.0, 10)
		train_sizes_abs, train_scores, valid_scores = model_selection.learning_curve(boost_model,
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
		plt.title("Adaboost learning curve 1 (n_estimators = 1)")
		plt.grid()
		plt.show()
		
		# classification report
		t0_clock = time.process_time()
		boost_model.fit(self.train_X, self.train_y)
		pred = boost_model.predict(self.test_X)  # Predict with test set
		t1_clock = time.process_time()
		print("The training time for final selected model is " + str(t1_clock - t0_clock) + " seconds")
		print(metrics.classification_report(self.test_y, pred, digits=4))


if __name__ == '__main__':
	data = pd.read_csv("../winequality.csv")
	label = "label"
	AdaBoost = AdaBoostAnalysis(data, label)
	
	AdaBoost.splitting_data()
	AdaBoost.estimator_pruning()
	AdaBoost.GridSearch_estimators()
	AdaBoost.n_estimators_validation()
	AdaBoost.GridSearch_with_algorithm()
	AdaBoost.learning_curve_and_output()
