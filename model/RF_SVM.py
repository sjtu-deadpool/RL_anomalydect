import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

X= pd.read_csv("goodFeatures.csv")
y= pd.read_csv("goodLabels.csv")
X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        train_size=0.7,
                                                        test_size=0.3,
                                                        random_state=0,
                                                        stratify=y)


svm = SVC(kernel='rbf', C=20.0, gamma='auto')
svm.fit(X_train, y_train)
y_pred = svm.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d",cmap='RdBu_r',center=300)

plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')

# # show the plot
plt.show()

# # print the confusion matrix
print(cm)

