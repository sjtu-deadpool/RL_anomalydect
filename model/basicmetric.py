# caculate the basic metrics of the model

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
import numpy as np

# generate random data
X = np.random.rand(100, 10)
y = np.random.randint(2, size=100)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = DecisionTreeClassifier()

def calMetrics(X_train,y_train,X_test,y_test,y_pred):
    # clf.fit(X_train, y_train)

    # y_pred = clf.predict(X_test)

    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = conf_matrix.ravel()
    false_alarm_rate = fp / (fp + tn)

    print("Precision:", precision)
    print("Recall:", recall)
    print("F1-score:", f1)
    print("Accuracy:", accuracy)
    print("False Alarm Rate:", false_alarm_rate)
    return precision,recall,f1,accuracy,false_alarm_rate
