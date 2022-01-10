# Quantum Reinforcement Learning applied to Board Games
In this project, we combined quantum computing with reinforcement learning and studied its application to a board game to assess the benefits that it can introduce, namely its impact on the learning efficiency of an agent.

We developed and applied a quantum-enhanced exploration policy to an off-policy RL algorithm, taking advantage of the state superposition principle and the amplitude amplification technique to deal with the convergence and the huge dimension of the state space. The game of Checkers was chosen due to its simple rules
yet average computational complexity, allowing us to focus on the learning process while presenting an interesting problem to apply RL.

## Background
Reinforcement learning is a machine learning paradigm where an agent learns how to optimize its behavior solely through its interaction with the environment. It has been extensively studied and successfully applied to complex problems of many different domains in the past decades, i.e., robotics, games, scheduling. However, the performance of these algorithms becomes limited as the complexity and dimension of the state-action space increases. 

Recent advances in quantum computing and quantum information have sparked interest in possible applications to machine learning. By taking advantage of quantum mechanics, it is possible to efficiently process immense quantities of information and improve computational speed.

## Conclusions
From the results, we concluded that the proposed quantum exploration policy improved the convergence rate of the agent and promoted a more efficient exploration of the state space.

## Improvements to be done
First, some aspects of the proposed exploration policy were not explicitly evaluated due to the lack of collected metrics, such as the number of iterations required to obtain a flagged action. Analyzing the differences in those aspects would allow us to better compare the classical and quantum approaches. 

Second, it would also be interesting to study the learning process of the agents when playing against human players. Unlike the non-human opponents tested in our work, a human player is constantly changing and improving its strategies, allowing us to further test the agent’s adaptability to changes in the environment and their ability of generalization. 

Finally, another interesting experiment would be to test both approaches in a more complex board game, e.g., Chess, and study how their performance scales in problems with larger state and action spaces.

Please share with us any improvement made.

## Authors
Main Author: Miguel Teixeira (up201605150@fe.up.pt)

Supervisors: Ana Paula Rocha (arocha@fe.up.pt) and António J. M. Castro (frisky.antonio@gmail.com)

## Project file structure:
The root of the project comprises four folders and 9 files:
* agents - software agents developed in python
* docs - contains the MSc Thesis Report and the WI-IAT 2021 paper with details about this work
* old_agents - some older versions of the software agents
* stats - some results and statistics from running experments

root files:
* notebook.ipynb - start with this notebokk
* Visualization.ipynb - notebook that shows the results of experiments
* several python files that implements parts of the code

## Run current code
* start with notebook.ipynb
