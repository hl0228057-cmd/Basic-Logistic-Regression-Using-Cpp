#include <iostream>
#include <vector>
#include <cmath>
using namespace std;

vector<double> inputs, expectedOutputs;

double sigmoidFunction(double z) {
    // 1 / (1 + e^(-z))
    return 1 / (1 + exp(-z));
}

double trainAndReturnZ(int n,  double newValue) {
    double bias = 0, weight = 0;
    int iterations = 3e+4;
    double learningRate = 1e-2;

    double currentLoss = 0;

    // initial loss
    for (int j = 0; j < n; j++) {
        double pred = sigmoidFunction(weight * inputs[j] + bias);
        // binary cross entropy punishes confident, incorrect answers heavily
        currentLoss += -(expectedOutputs[j] * log(pred) + (1 - expectedOutputs[j]) * log(1 - pred));
    }
    currentLoss /= n;

    // training loop
    for (int i = 0; i < iterations; i++) {
        if (currentLoss < 1e-6) break;

        double biasChange = 0, weightChange = 0;
        currentLoss = 0;

        for (int j = 0; j < n; j++) {
            double pred = sigmoidFunction(weight * inputs[j] + bias);

            // binary cross entropy punishes confident, incorrect answers heavily
            currentLoss += -(expectedOutputs[j] * log(pred) + (1 - expectedOutputs[j]) * log(1 - pred));

            double error = pred - expectedOutputs[j];
            biasChange += error;
            weightChange += error * inputs[j];
        }

        currentLoss /= n;
        biasChange /= n;
        weightChange /= n;

        bias -= learningRate * biasChange;
        weight -= learningRate * weightChange;

        // learning rate decay
        if (i > 0 && i % 1000 == 0) learningRate *= 0.95;
    }

    // z = wx + b
    return weight * newValue + bias;
}

int main() {
    // data
    int dataLength = 0;
    while (dataLength < 2) {
        cout << "number of data points: ";
        cin >> dataLength;
    }

    const int n = dataLength;

    inputs.resize(n);
    expectedOutputs.resize(n);

    for (int i = 0; i < n; i++) {
        cout << "time studying (hours): "; cin >> inputs[i];
        cout << "passed? (0 for failed, 1 for passed): "; cin >> expectedOutputs[i];
    }

    double nextValue; cout << "time studied for prediction (hours): "; cin >> nextValue;

    double z = trainAndReturnZ(n, nextValue);
    double pred = sigmoidFunction(z);
    double thresholdToPass = 0.5;

    if (pred >= thresholdToPass) {
        cout << "pass; probability of passing: " << pred * 100 << '%';
    } else {
        cout << "fail; probability of failing: " << (1 - pred) * 100 << '%';
    }

    return 0;
}