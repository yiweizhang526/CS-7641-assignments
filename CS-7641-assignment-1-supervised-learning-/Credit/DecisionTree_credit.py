import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt

from pandas.api.types import is_string_dtype
from pandas.api.types import is_categorical_dtype

from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn import model_selection


class DecisionTreeAnalysis:
	def __init__(self, data, label):
		self.data = data
		self.label = label
		self.seed = 166
		self.cv = 5
		self.n_jobs = -1
		self.alpha = list()
		
		self.train_X, self.train_y = [], []
		self.test_X, self.test_y = [], []
	
	def splitting_data(self):
		predictors = self.data[self.data.columns.difference([label])]
		target = self.data[[label]]
		
		self.train_X, self.test_X, self.train_y, self.test_y = model_selection.train_test_split \
			(predictors, target, train_size=0.9, random_state=self.seed, stratify=target)
		
		treeModel = DecisionTreeClassifier(random_state=self.seed)
		treeModel.fit(self.train_X, self.train_y)
		pred = treeModel.predict(self.test_X)
		print("f1: " + str(metrics.f1_score(self.test_y, pred)))
	
	def prior_pruning(self):
		'''
		prior-clipping: searching for suitable max-depth
		'''
		tree_model = DecisionTreeClassifier(criterion="entropy", random_state=self.seed)
		max_depth = np.arange(1, 62, 1)
		train_scores, valid_scores = model_selection.validation_curve \
			(tree_model, self.train_X, self.train_y, param_name="max_depth", param_range=max_depth, cv=self.cv,
			 n_jobs=self.n_jobs, scoring="f1")
		
		plt.figure()
		plt.plot(max_depth, train_scores.mean(axis=1), 'r', label="training score")
		plt.plot(max_depth, valid_scores.mean(axis=1), 'g', label="cross-validation score")
		plt.xlabel('Max depth')
		plt.ylabel('f1 Score')
		plt.legend(loc="best")
		plt.title("Decision tree prior_pruning experiments: max_depth")
		plt.grid()
		plt.show()
		
	def post_pruning(self):
		'''
		searching for ccp_alpha
		'''
		tree_model = DecisionTreeClassifier(random_state=self.seed)
		path = tree_model.cost_complexity_pruning_path(self.train_X, self.train_y)
		ccp_alphas, impurities = path.ccp_alphas, path.impurities
		
		train_scores, valid_scores = model_selection.validation_curve \
			(tree_model, self.train_X, self.train_y, param_name="ccp_alpha", param_range=ccp_alphas[:-1], cv=self.cv,
			 n_jobs=self.n_jobs, scoring="f1")
		
		plt.figure()
		plt.plot(ccp_alphas[:-1], train_scores.mean(axis=1), 'o-', color='r', label="training score")
		plt.plot(ccp_alphas[:-1], valid_scores.mean(axis=1), 'o-', color='g', label="cross-validation score")
		plt.xlabel('ccp alpha')
		plt.ylabel('f1 Score')
		plt.legend(loc="best")
		plt.title("Decision tree Post_pruning experiments: ccp_alpha")
		plt.grid()
		plt.show()
		
		for ccp_alpha in ccp_alphas:
			if 0.002 < ccp_alpha < 0.004:
				self.alpha.append(ccp_alpha)
	
	def GridSearch_with_criterion(self):
		'''
		max_depth, criterion, ccp_alpha
		'''
		tree_model = DecisionTreeClassifier(random_state=self.seed)
		tuned_parameters = {'max_depth': range(10, 20),
							'criterion': ["gini", "entropy"],
							'ccp_alpha': self.alpha}
		clf = model_selection.GridSearchCV(tree_model, tuned_parameters, scoring="f1", n_jobs=self.n_jobs,
										   cv=self.cv)
		clf.fit(self.train_X, self.train_y)
		print(clf.best_score_, clf.best_params_)
	
	def learning_curve_and_output(self):
		'''
		test the best model
		'''
		tree_model = DecisionTreeClassifier(criterion="entropy", max_depth=14, ccp_alpha=0.002, random_state=self.seed)
		# tree_model = DecisionTreeClassifier(random_state=self.seed)
		train_sizes = np.linspace(0.1, 1.0, 10)
		train_sizes_abs, train_scores, valid_scores = model_selection.learning_curve(tree_model,
																					self.train_X, self.train_y,
																					cv=self.cv, n_jobs=self.n_jobs,
																					train_sizes=train_sizes,
																					random_state=self.seed,
																					scoring="f1")

		plt.figure()
		plt.plot(train_sizes_abs, train_scores.mean(axis=1), 'o-', color='r', label="training score")
		plt.plot(train_sizes_abs, valid_scores.mean(axis=1), 'o-', color='g', label="cross-validation score")
		plt.xlabel('Size of training set')
		plt.ylabel('f1 Score')
		plt.legend(loc="best")
		plt.title("Decision tree learning curve")
		plt.grid()
		plt.show()
		
		# classification report
		t0_clock = time.process_time()
		tree_model.fit(self.train_X, self.train_y)
		pred = tree_model.predict(self.test_X)  # Predict with test set
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
	DecisionTree = DecisionTreeAnalysis(data, label)
	
	DecisionTree.splitting_data()
	DecisionTree.prior_pruning()
	DecisionTree.post_pruning()
	DecisionTree.GridSearch_with_criterion()
	DecisionTree.learning_curve_and_output()
