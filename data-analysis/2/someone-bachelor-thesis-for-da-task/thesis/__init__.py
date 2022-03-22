from sklearn.model_selection import KFold, cross_val_score

import numpy as np
import matplotlib.pyplot as plt
import time
import datetime
from typing import *
import yaml

columns = ("classifier", "False positive error", "True negative error", "False negative", "True positive", "TIME")

def reload_config():
	global config
	with open("conf.yaml") as stream:
		config = yaml.safe_load(stream)

reload_config()

seed : int = config['seed']

class Worker:
	def __init__(self, xtr, xtest, ytr, ytest, rng : Iterable[Any], supplier : Callable[[int], Any]):
		self.X_train = xtr
		self.X_test = xtest
		self.y_train = ytr
		self.y_test = ytest
		self.rng = rng
		self.supplier = supplier

	def calc_accuracy(self) -> None:
		accuracy = []
		best_k = 1
		best_score = 0
		for k in self.rng:
			classifier = self.supplier(k)
			# kf = KFold(len(y_train), random_state=70, n_folds=5,shuffle=True)
			kf = KFold(n_splits=5, random_state=seed, shuffle=True)
			scores = cross_val_score(classifier, self.X_train, self.y_train, cv=kf)
			if scores.mean() > best_score:
				best_score = scores.mean()
				best_k = k
			accuracy.append(scores.mean())
		self.accuracy, self.best_k, self.best_score = (accuracy, best_k, best_score)

	def plot_accuracy(self) -> None:
		plt.plot(self.rng, self.accuracy)
		plt.show()
		print ("Best k = ", self.best_k)
		print ("Best score = ", self.best_score)

	def calc_and_print(self, name : str = None) -> Tuple[str, float, float, float, float, Any]:
		accuracy, best_k, best_score = (self.accuracy, self.best_k, self.best_score)
		y_test = self.y_test
		start_time = datetime.datetime.now()
		knn = self.supplier(best_k)
		knn.fit(self.X_train, self.y_train)
		y_test_predict = knn.predict(self.X_test)
		fp = 0
		tn = 0
		fn = 0
		tp = 0
		f = 0
		t = 0
		for i in range(len(y_test)):
			if y_test[i] == 0 and y_test_predict[i] == 1:
				fp += 1
				f += 1
			if y_test[i] == 0 and y_test_predict[i] == 0:
				fn += 1
				f += 1
			else:
				if y_test[i] == 1 and y_test_predict[i] == 0:
					tn += 1
					t += 1
				if y_test[i] == 1 and y_test_predict[i] == 1:
					tp += 1
					t += 1

		params = (name, float(fp) / len(y_test), float(tn) / len(y_test), float(fn) / f, float(tp) / t, datetime.datetime.now() - start_time)
		for a, b in zip(columns, params):
			print(a, "\t = ", b)
		return params

	def all_in_one(self, name: str = None) -> Tuple[str, float, float, float, float, Any]:
		self.calc_accuracy()
		self.plot_accuracy()
		return self.calc_and_print(name)
