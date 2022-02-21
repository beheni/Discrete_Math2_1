
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
iris = load_iris()

class Node:

    def __init__(self, X, y, gini=0, feature_index=0, threshold=0, left=None, right=None):
        self.X = X
        self.y = y
        self.gini = gini
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right


class MyDecisionTreeClassifier:

    def __init__(self, max_depth):
        self.max_depth = max_depth
        self.root = 0
    def gini(self, y):
        """
        classes = [[0],[1],[2]]
        A Gini score gives an idea of how good a split is by how mixed the
        classes are in the two groups created by the split.
        A perfect separation results in a Gini score of 0,
        whereas the worst case split that results in 50/50
        classes in each group result in a Gini score of 0.5
        (for a 2 class problem).
        """
        gini = 0
        all = len(y)
        for cls in range(3):
            sum_classes = []
            for i in range(all):
                clas = int(y[i])
                if clas == cls:
                    sum_classes.append(clas)
            p = len(sum_classes) / all
            gini += p ** 2
        gini = 1 - gini
        return gini


    def split_data(self, X, y):
        IG = [0, 0, 0, 0, 0, 0, 0, 0]
        num_features = iris.feature_names
        for feature_index in range(len(num_features)):
            possible_thresholds = sorted(np.unique(X[:, feature_index]), reverse=True)
            for threshold in possible_thresholds:
                data_left = []
                data_right = []
                y_left = []
                y_right = []
                for i in range(len(X)):
                    pi = X[i, feature_index]
                    if pi <= threshold:
                        data_left.append(X[i])
                        y_left.append(y[i])
                    else:
                        data_right.append(X[i])
                        y_right.append(y[i])
                data_parent = X
                data_left = np.array(data_left)
                data_right = np.array(data_right)
                y_left = np.array(y_left)
                y_right = np.array(y_right)
                if len(data_left) > 0 and len(data_right) > 0:
                    gini_p = self.gini(y)
                    gini_l = self.gini(y_left)
                    gini_r = self.gini(y_right)
                    weight_l = len(data_left) / len(data_parent)
                    weight_r = len(data_right) / len(data_parent)
                    ig = gini_p - (weight_l * gini_l + weight_r * gini_r)
                    if ig > IG[0]:
                        IG[0] = ig
                        IG[1] = data_left
                        IG[2] = data_right
                        IG[3] = data_parent
                        IG[4] = threshold
                        IG[5] = feature_index
                        IG[6] = y_left
                        IG[7] = y_right
        return IG

    def build_tree(self, X, y, depth=0, nodes=[], leaves=[]):
        if len(set(y)) > 1:
            if depth <= self.max_depth:
                split = self.split_data(X, y)
                if split[0] > 0:
                    nodes.append(Node(X, y, split[0], split[5], split[4]))
                    left_son = self.build_tree(split[1], split[6], depth + 1)
                    right_son = self.build_tree(split[2], split[7], depth + 1)
        else:
            nodes.append(Node(X=X, y=iris.target_names[int(y[0])]))
            leaves.append(Node(X=X, y=iris.target_names[int(y[0])]))
        return nodes

    def print_tree(self, X, y, indent=" "):
        self.root = self.build_tree(X, y)
        for node in self.root:
            print(f"{iris.feature_names[node.feature_index]} <= {node.threshold}" + '\n' + f'gini = {node.gini}' + '\n' + f"samples = {len(node.X)}" + '\n' + f"name {node.y}")

    def fit(self, X, y):
        # basically wrapper for build tree

        pass


my_tree = MyDecisionTreeClassifier(10)
print(my_tree.print_tree(iris.data[:-38], iris.target[:-38]))
