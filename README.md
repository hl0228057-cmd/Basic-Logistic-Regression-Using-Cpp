# Basic Logistic Regression Using C++

Description:
A simple logistic regression model implemented entirely from scratch in C++ without using any machine learning libraries. I made this project to better understand how classification-based machine learning works internally, including sigmoid activation, binary cross-entropy loss, gradient descent, probabilities, and parameter optimization.

Features: sigmoid function, binary cross-entropy loss, gradient descent, user-input training data, and binary classification

How it works:
In the training loop, the model repeatedly: computes a linear combination (z = wx + b), converts the raw score into a probability using sigmoid, calculates prediction error, computes binary cross-entropy loss, updates weights and bias using gradient descent

Example:
  Input:
    1 -> 0,
    2 -> 0,
    3 -> 1,
    4 -> 1,
  Prediction input:
    2.5
  Output:
    pass; probability of passing: 55.978%

After completing this project, I learned how logistic regression work.
