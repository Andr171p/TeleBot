from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier


class Classifier:
    classifier: RandomForestClassifier = None
    # classifier: LogisticRegression = None
    # classifier: DecisionTreeClassifier = None