import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt

from pandas.api.types import is_string_dtype
from pandas.api.types import is_categorical_dtype

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn import metrics
from sklearn import model_selection


class AdaBoostAnalysis:
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
		target = self.data[[self.label]]
		
		self.train_X, self.test_X, self.train_y, self.test_y = model_selection.train_test_split \
			(predictors, target, train_size=0.9, random_state=self.seed, stratify=target)
	
		boost_model = AdaBoostClassifier(random_state=166)
		boost_model.fit(self.train_X, self.train_y)
		pred = boost_model.predict(self.test_X)
		print("f1: " + str(metrics.f1_score(self.test_y, pred)))
	
	def estimator_pruning(self):
		'''
		searching for min_samples_split, min_samples_leaf
		'''
		tree_model = DecisionTreeClassifier(criterion="entropy", max_depth=14, random_state=self.seed)
		min_samples_split = np.arange(2, 21, 1)
		train_scores1, valid_scores1 = model_selection.validation_curve \
			(tree_model, self.train_X, self.train_y, param_name="min_samples_split", param_range=min_samples_split,
			 cv=self.cv, n_jobs=self.n_jobs, scoring="f1")
		min_samples_leaf = np.arange(1, 11, 1)
		train_scores2, valid_scores2 = model_selection.validation_curve \
			(tree_model, self.train_X, self.train_y, param_name="min_samples_leaf", param_range=min_samples_leaf,
			 cv=self.cv, n_jobs=self.n_jobs, scoring="f1")

		plt.figure(figsize=(6,9))
		plt.subplot(211)
		plt.plot(min_samples_split, train_scores1.mean(axis=1), 'r', label="training score")
		plt.plot(min_samples_split, valid_scores1.mean(axis=1), 'g', label="cross-validation score")
		plt.xlabel('Min samples split')
		plt.ylabel('f1 Score')
		plt.legend(loc="best")
		plt.title("Decision tree prior_pruning experiments: min samples split")
		plt.grid()
		plt.subplot(212)
		plt.plot(min_samples_leaf, train_scores2.mean(axis=1), 'r', label="training score")
		plt.plot(min_samples_leaf, valid_scores2.mean(axis=1), 'g', label="cross-validation score")
		plt.xlabel('Min samples leaf')
		plt.ylabel('f1 Score')
		plt.legend(loc="best")
		plt.title("Decision tree prior_pruning experiments: min samples leaf")
		plt.grid()
		plt.show()
	
	def GridSearch_estimators(self):
		'''
		max_depth, criterion, min_samples_leaf, min_samples_split
		'''
		tree_model = DecisionTreeClassifier(random_state=self.seed)
		tuned_parameters = {'max_depth': range(10, 20),
							'criterion': ["gini", "entropy"],
							'min_samples_split': range(2, 13),
							'min_samples_leaf': range(1, 7)}
		clf = model_selection.GridSearchCV(tree_model, tuned_parameters, scoring="f1", n_jobs=self.n_jobs,
										   cv=self.cv)
		clf.fit(self.train_X, self.train_y)
		print(clf.best_score_, clf.best_params_)
	
	def n_estimators_validation(self):
		'''
		searching for suitable n_estimators
		'''
		tree_model = DecisionTreeClassifier(max_depth=14, criterion="entropy", min_samples_leaf=5, min_samples_split=11,
											random_state=self.seed)
		boost_model = AdaBoostClassifier(base_estimator=tree_model, random_state=self.seed)
		
		n_estimators = np.arange(1, 100, 2)
		train_scores, valid_scores = model_selection.validation_curve \
			(boost_model, self.train_X, self.train_y, param_name="n_estimators", param_range=n_estimators, cv=self.cv,
			 n_jobs=self.n_jobs, scoring="f1")
		
		plt.figure()
		plt.plot(n_estimators, train_scores.mean(axis=1), 'o-', color='r', label="training score")
		plt.plot(n_estimators, valid_scores.mean(axis=1), 'o-', color='g', label="cross-validation score")
		plt.xlabel('n estimators')
		plt.ylabel('f1 Score')
		plt.legend(loc="best")
		plt.title("Adaboost pruning experiments: n_estimators (Decision Tree)")
		plt.grid()
		plt.show()
	
	def GridSearch_with_algorithm(self):
		'''
		n_estimators, algorithm
		'''
		tree_model = DecisionTreeClassifier(max_depth=14, criterion="entropy", min_samples_leaf=5, min_samples_split=11,
											random_state=self.seed)
		boost_model = AdaBoostClassifier(base_estimator=tree_model, random_state=self.seed)
		
		tuned_parameters = {'n_estimators': range(10, 40), 'algorithm': ["SAMME", "SAMME.R"]}
		clf = model_selection.GridSearchCV(boost_model, tuned_parameters, scoring="f1", n_jobs=self.n_jobs,
										   cv=self.cv)
		clf.fit(self.train_X, self.train_y)
		print(clf.best_score_, clf.best_params_)
	
	def learning_curve(self):
		tree_model = DecisionTreeClassifier(max_depth=14, criterion="entropy", min_samples_leaf=5, min_samples_split=11,
											)
		boost_model = AdaBoostClassifier(base_estimator=tree_model, n_estimators=36, algorithm="SAMME", random_state=166)
		
		train_sizes = np.linspace(0.1, 1.0, 10)
		train_sizes_abs, train_scores, valid_scores = model_selection.learning_curve(boost_model,
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
		plt.title("Adaboost learning curve 1 (n_estimators = 36)")
		plt.grid()
		plt.show()

	def max_depth_tuning(self):
		'''
		works only in boosting method
		'''
		scores = list()
		for k in range(1, 14):
			tree_model = DecisionTreeClassifier(max_depth=k, criterion="entropy", min_samples_leaf=5, min_samples_split=11,
												random_state=self.seed)
			boost_model = AdaBoostClassifier(base_estimator=tree_model, n_estimators=36, algorithm="SAMME", random_state=self.seed)
			boost_model.fit(self.train_X, self.train_y)
			pred = boost_model.predict(self.test_X)
			scores.append(metrics.f1_score(self.test_y, pred))
		
		plt.figure(figsize=(6,4))
		plt.plot(range(1, 14), scores, 'o-', color='r', label="testing score")
		plt.xlabel('max_depth of base classifier (Decision Tree)')
		plt.ylabel('f1 score')
		plt.legend(loc="best")
		plt.title("Adaboost: test score")
		plt.grid()
		plt.show()
	
	def compare_and_output(self):
		'''
		test the best model
		'''
		tree_model = DecisionTreeClassifier(max_depth=2, criterion="entropy", min_samples_leaf=5, min_samples_split=11,
											random_state=self.seed)
		boost_model = AdaBoostClassifier(base_estimator=tree_model, n_estimators=36, algorithm="SAMME",
										 random_state=self.seed)
		# classification report
		t0_clock = time.process_time()
		boost_model.fit(self.train_X, self.train_y)
		pred = boost_model.predict(self.test_X)  # Predict with test set
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
	AdaBoost = AdaBoostAnalysis(data, label)
	
	AdaBoost.splitting_data()
	AdaBoost.estimator_pruning()
	AdaBoost.GridSearch_estimators()
	AdaBoost.n_estimators_validation()
	AdaBoost.GridSearch_with_algorithm()
	AdaBoost.learning_curve()
	AdaBoost.compare_and_output()
