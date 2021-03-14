# Offline reinforcement learning for neural architecture search

A project lead by Martin Cepeda, Romain Egele and Jeremy Gozlan during the reinforcement learning lecture
of the AIViC master at Ecole polytechnique.

The repository is composed of an RL environment `env/nas_rl.py`.

The `data_generation.py`  was used to generate our artificial datasets.

The different notebooks are:

* `Classic RL Environment`: a regular environment for NAS with on-policy and off-policy RL.
* `Offline RL Environment`: a environment for NAS with offline RL.
* `Supervised learning on the Dataset`: a context-based bandit approach for NAS.
* `Offline DQN`: a DQN algorithm adapted to the offline RL setting.