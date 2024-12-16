import random
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, balanced_accuracy_score
import basicmetric
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
X= pd.read_csv("goodFeatures.csv")
y= pd.read_csv("goodLabels.csv")
X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        train_size=0.7,
                                                        test_size=0.3,
                                                        random_state=0,
                                                        stratify=y)
from pso_svm import utils
# from utils import data_handle_v1, data_handle_v2
# from config.config import args, kernel, data_src, data_path
# X_train, X_test, y_train, y_test = utils.data_handle_v1("D:/private/anomalydetection/model/pso_svm/data/Statlog_heart_Data.csv")

import time

# start time
start_time = time.time()

import numpy as np
import math
# result = math.e ** math.pi
# initialize Q table, state space and action space
# here we assume the state space is [0.1, 0.2, ..., 50] and the action space is {-1, 0, 1}
num_states = 10
num_actions = 3
Q = np.zeros((num_states, num_actions))

# define state space and action space
state_space = np.linspace(0.1, 50, num_states)
action_space = np.array([-1, 0, 1])  # action space is {-1, 0, 1}

# define Q-learning parameters
learning_rate = 0.1
discount_factor = 0.9
num_episodes = 10  # iteration number
exploration_rate=0.5
state = state_space[4]

y_pred_best=[]
best_C=0.1
bestModel = None
for episode in range(num_episodes):
    # print("episode:",episode)
    for step in range(10):
        if np.random.uniform(0,1) < exploration_rate or state not in Q:
            # action_idx = np.random.randint(1, 10)-1  # randomly select C value
            action = np.random.choice(action_space)
            action_index = np.where(action_space == action)[0][0]
        else:
            # action_idx = np.argmax(Q[state_space == state, :])
            # choose the action with the highest Q value
            state_index = np.where(state_space == state)[0][0]
            action_index = np.argmax(Q[state_index, :])
            action = action_space[action_index]

        # update state
        if action == 1:
            new_state_idx = np.argmin(np.abs(state_space - state)) + 1 
        elif action == -1:
            new_state_idx = np.argmin(np.abs(state_space - state)) - 1 
        else:
            new_state_idx = np.argmin(np.abs(state_space - state)) 

        # ensure the new state is within the state space
        new_state_idx = max(0, min(new_state_idx, num_states - 1))

        # obtain the new state
        new_state = state_space[new_state_idx]


        # update Q-table
        # get reward
        clf = svm.SVC(kernel='rbf',C=new_state,gamma=0.95)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        precision,recall,f1,accuracy,false_alarm_rate=basicmetric.calMetrics(X_train,y_train,X_test,y_test,y_pred)
        # scores = cross_val_score(clf, X, y, cv=5)
        score=0.2*precision+0.2*recall+0.2*f1+0.2*accuracy+0.2*math.e **false_alarm_rate
        reward = np.mean(score)
        
        # calculate the maximum Q value for the new state
        max_q_new_state = np.max(Q[new_state_idx, :])

        # update Q value
        state_index = np.where(state_space == state)[0][0]
        Q[state_index, action_index] += learning_rate * (reward + discount_factor * max_q_new_state - Q[state_index, action_index])
        if Q[state_index, action_index]==np.argmax(Q):
            y_pred_best=y_pred
            best_C=state
            bestModel=clf
        print(state,action,Q[state_index, action_index])
        state = new_state

# get the best C value
best_q_idx = np.unravel_index(np.argmax(Q), Q.shape)
best_c = state_space[best_q_idx[0]]

print("Best C:", best_c)
cm = confusion_matrix(y_test, y_pred_best)
sns.heatmap(cm, annot=True, fmt="d",cmap='RdBu_r',center=300)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()
# plt.savefig("qsvm_hotmap.png")

end_time = time.time()

# caculate the training time
training_time = end_time - start_time
print("training time", training_time, "seconds")
print(Q)