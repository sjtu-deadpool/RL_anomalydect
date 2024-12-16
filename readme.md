# ELEN E6885 yz4895

## Final Project: Reinforcement Learning Based Hyperparameter Optimisationof Support Vector Machines for Anomalous Traffic Detection

Anomaly detection in network traffic is a critical task for ensuring cybersecurity, yet traditional intrusion detection systems often face challenges in feature selection and hyperparameter optimization. This project leverages reinforcement learning (RL) to automate the tuning of hyperparameters in Support Vector Machines (SVMs), specifically focusing on the penalty parameter (C) and the kernel coefficient (gamma). A two-phase approach is proposed: first, Random Forests are employed for feature selection, isolating the most relevant subset of features; second, a Q-learning-based RL algorithm dynamically adjusts the SVM hyperparameters to maximize detection performance. Experiments are conducted on benchmark datasets, including CICIDS2019, CICIDS2017, and NSL-KDD, with comparisons against traditional optimization techniques such as Particle Swarm Optimization and Genetic Algorithms. Results demonstrate that the RL-based optimization approach significantly improves detection accuracy and convergence speed. Additionally, its application in Software-Defined Network (SDN) environments highlights its potential for real-world anomaly detection scenarios.

First prepare the running environment:
```bash
conda create -n ELEN6885 python=3.12
conda activate ELEN6885
# install required packages
pip install -r requirements.txt
```

Data acquisition and processing scripts are available at `/data`.

Code for model runs and comparisons in `/model`.