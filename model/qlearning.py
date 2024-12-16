import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import RF_feature
fromPath = "dataset.csv"
LabelColumnName = ' Label'
dataset, featureList = preprocess.loadData(fromPath, LabelColumnName, 2)
X, y = RF_feature.preprocess()

X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        train_size=0.7,
                                                        test_size=0.3,
                                                        random_state=0,
                                                        stratify=y)



# Q-learning algorithm
class QLearning:
    def __init__(self, state_space, action_space, learning_rate=0.1, discount_factor=0.9):
        self.state_space = state_space
        self.action_space = action_space
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.q_table = np.zeros((len(state_space), len(action_space)))

    def choose_action(self, state):
        state_index = self.state_space.index(state)
        q_values = self.q_table[state_index, :]
        max_q_value = np.max(q_values)
        max_action_indices = np.where(q_values == max_q_value)[0]
        chosen_action_index = np.random.choice(max_action_indices)
        chosen_action = self.action_space[chosen_action_index]
        return chosen_action

    def update_q_table(self, state, action, next_state, reward):
        state_index = self.state_space.index(state)
        next_state_index = self.state_space.index(next_state)
        action_index = self.action_space.index(action)
        max_q_value = np.max(self.q_table[next_state_index, :])
        current_q_value = self.q_table[state_index, action_index]
        new_q_value = (1 - self.learning_rate) * current_q_value + self.learning_rate * (reward + self.discount_factor * max_q_value)
        self.q_table[state_index, action_index] = new_q_value
    
    # def reward()
    
# define a function to select features
def select_features(X, y, feature_combination):
    selected_X = X[:, feature_combination]
    clf = DecisionTreeClassifier()
    clf.fit(selected_X, y)
    return clf

# define a function to detect anomalies
def detect_anomalies(q_learning, X, y):
    num_features = X.shape[1]
    selected_features = []
    for i in range(num_features):
        state = tuple(selected_features)
        action = q_learning.choose_action(state)
        selected_features.append(action)

        clf = select_features(X, y, selected_features)
        predictions = clf.predict(X[:, selected_features])

        accuracy = np.mean(predictions == y)
        reward = 2 * accuracy - 1

        next_state = tuple(selected_features)
        q_learning.update_q_table(state, action, next_state, reward)

    return selected_features

# example data
# X = np.array([[1, 2, 3, 4],
#               [2, 3, 4, 5],
#               [3, 4, 5, 6],
#               [4, 5, 6, 7]])
# y = np.array([0, 0, 1, 1])

rows = 10
columns = 79
q_table = np.zeros((rows, columns))

matrix = [[0 for _ in range(79)] for _ in range(10)]
state_space = np.zeros((10, 10))          
action_space = list(range(1, 80))

q_learning = QLearning(state_space, action_space)
selected_features = detect_anomalies(q_learning, X, y)

print("Selected Features:", selected_features)
