import random
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, balanced_accuracy_score
# import basicmetric
from sklearn.cluster import KMeans
# from sklearn.metrics import silhouette_score
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

X= pd.read_csv("./goodFeatures.csv")
y= pd.read_csv("./goodLabels.csv")
X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        train_size=0.7,
                                                        test_size=0.3,
                                                        random_state=0,
                                                        stratify=y)



import time

start_time = time.time()

import numpy as np

num_states = 10
num_actions = 3
Q = np.zeros((num_states, num_actions))

# state_space = np.linspace(0.1, 50, num_states) 
state_space = np.arange(2, 11)  

action_space = np.array([-1, 0, 1]) 

learning_rate = 0.1
discount_factor = 0.9
num_episodes = 100 
exploration_rate=0.5
state = 5
for episode in range(num_episodes):
    print("episode:",episode)

    # action_idx = np.argmax(Q[state_space == state, :])  
    
    # action = action_space[action_idx]
    if np.random.uniform(0,1) < exploration_rate or state not in Q:
        # action_idx = np.random.randint(1, 10)-1  # randomly select C value
        action = np.random.choice(action_space)
        action_index = np.where(action_space == action)[0][0]
    else:
        # action_idx = np.argmax(Q[state_space == state, :])
        state_index = np.where(state_space == state)[0][0]
        action_index = np.argmax(Q[state_index, :])
        action = action_space[action_index]
    # action = action_space[action_idx]

    if action == 1:
        new_state_idx = np.argmin(np.abs(state_space - state)) + 1 
    elif action == -1:
        new_state_idx = np.argmin(np.abs(state_space - state)) - 1 
    else:
        new_state_idx = np.argmin(np.abs(state_space - state)) 

    new_state_idx = max(0, min(new_state_idx, num_states - 1))

    new_state = state_space[new_state_idx]


    # clf = svm.SVC(kernel='rbf',C=new_state,gamma=0.95)
    # precision,recall,f1,accuracy,false_alarm_rate=basicmetric.calMetrics(clf,X_train,y_train,X_test,y_test)
    # # scores = cross_val_score(clf, X, y, cv=5)
    # score=0.25*precision+0.25*recall+0.25*f1+0.25*accuracy+0.25*false_alarm_rate
    # reward = np.mean(score)
    kmeans = KMeans(n_clusters=new_state).fit(X_train)
    labels = kmeans.labels_
    inertia = kmeans.inertia_
    silhouette = silhouette_score(X_train, labels)
    calinski_harabasz = calinski_harabasz_score(X_train, labels)
    davies_bouldin = davies_bouldin_score(X_train, labels)

    print("Inertia:", inertia)
    print("Silhouette Score:", silhouette)
    print("Calinski-Harabasz Score:", calinski_harabasz)
    print("Davies-Bouldin Score:", davies_bouldin)
    reward = silhouette_score(X_train, labels)

    max_q_new_state = np.max(Q[new_state_idx, :])

    state_index = np.where(state_space == state)[0][0]
    Q[state_index, action_index] += learning_rate * (reward + discount_factor * max_q_new_state - Q[state_index, action_index])
    print(state,action,Q[state_index, action_index])
    state = new_state

best_k_idx = np.argmax(Q[:, :])
best_k = state_space[best_k_idx]
print("Best K:", best_k)


end_time = time.time()

training_time = end_time - start_time
print("training time:", training_time, "seconds")