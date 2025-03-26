# Machine Learning Homework: Decision Tree Analysis

## Project Overview
This homework focuses on analyzing a decision tree through both theoretical (pen-and-paper) and practical (programming) approaches using the `pd_speech.arff` dataset.

## Theoretical Tasks
The theoretical component involves:
- Drawing a training confusion matrix
- Identifying training F1 score after post-pruning the decision tree to a maximum depth of 1
- Exploring reasons for specific tree path decomposition limitations
- Computing information gain for a specific variable (y1)

## Programming Implementation
The programming task requires:
- Using scikit-learn to perform a stratified 70-30 train-test split
- Applying feature selection based on mutual information
- Training decision trees with varying numbers of selected features (5, 10, 40, 100, 250, 700)
- Creating a visualization comparing training and testing accuracies
- Analyzing why training accuracy consistently reaches 1

## Key Technical Components
- Machine Learning Technique: Decision Tree
- Feature Selection Method: Mutual Information
- Data Split: Stratified 70-30 Train-Test Split
- Libraries: Scikit-learn
- Visualization: Single plot showing training and testing accuracies

## Objectives
- Understand decision tree learning mechanics
- Explore feature selection impact on model performance
- Analyze model accuracy and potential overfitting
