# Linguistic Roots Classifier: Logistic Regression from First Principles

## Overview
This repository contains a Machine Learning project built entirely from scratch using pure NumPy. The objective is to classify the historical roots of technical English words—specifically distinguishing between everyday **Germanic** (Class 0) and complex **Greco-Latin** (Class 1) words based on their structural features.

Instead of importing black-box libraries like `scikit-learn`, I wanted to understand the absolute foundational logic of optimization theory and matrix calculus. Every component of this pipeline, from the cost function to feature normalization, is written from first principles.

## The Mathematical Engine
- **Vectorized Gradient Descent:** Implemented the gradient descent loop using pure matrix broadcasting (`dw = (1/m) * X.T · (A - Y)`), completely eliminating inefficient `for` loops.
- **Strictly Convex Optimization:** Derived and implemented the Binary Cross-Entropy (Log Loss) cost function to ensure the engine navigates a strictly convex space to a global minimum.
- **Custom Standardization:** Built a Z-score normalization pipeline from scratch to handle polynomial features and prevent Float64 hardware overflow during exponential operations.
- **Engineered Features:** Word length, vowel-to-consonant ratios, and custom consonant/suffix flags.

## The Data Discovery (Math > Data)
During error analysis, the engine's decision boundary revealed a dense cluster of "False Positives." The model assigned $>0.90$ probabilities to words like *action*, *active*, and *absence*, confidently flagging them as Greco-Latin. 

However, the open-source dataset labeled these as Germanic. 

By pulling the vectors and analyzing the history, I discovered the dataset was corrupted. It had historically misclassified thousands of Latin words introduced during the Middle English period as Anglo-Saxon. The gradient descent algorithm had successfully isolated the Latin structural weights and classified them correctly, defying the flawed ground-truth labels. The math proved smarter than the data.

## Visualizations
The notebook includes custom data visualizations built with `matplotlib`:
1. **The Learning Curve:** A plot of the Cost Function $J(w, b)$ proving the convergence of the gradient descent algorithm.
2. **Model Confidence Boundary:** A scatter plot utilizing calculated probabilistic jitter to visualize the $0.5$ decision boundary and highlight the model's predictive density. 

*(Note: If GitHub fails to render the heavy visualizations in the `.ipynb` file, copy the repository URL into [nbviewer.jupyter.org](https://nbviewer.jupyter.org/) for instant rendering.)*

## Next Steps & Ongoing Learning
Manual feature engineering (like Regex) is rigid and struggles with historical exceptions. As I continue exploring AI, the next iteration of this project will move away from manual feature flags and implement algorithmic **N-Gram vectorization** to mathematically deduce sub-word structures. 

## Requirements
- `numpy`
- `pandas`
- `matplotlib`
