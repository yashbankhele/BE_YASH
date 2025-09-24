🧠 Machine Learning Practicals – AI & Data Science (BE, 7th Semester)

This repository contains practical implementations of Machine Learning concepts covering Feature Transformation, Regression, Classification, Clustering, Ensemble Learning, and Reinforcement Learning. Each practical is implemented in Python using libraries like pandas, numpy, scikit-learn, matplotlib, and seaborn.

1️⃣ Feature Transformation – PCA

🎯 Objective:
Reduce dimensionality of a wine dataset to capture maximum variance and make it easier to distinguish between red and white wines.

📂 Dataset: Wine Dataset

🛠 Implementation:

Standardized numerical features.

Applied Principal Component Analysis (PCA).

Visualized first 2 principal components to separate wine types.

✅ Outcome:
Reduced feature space preserves most variance and simplifies analysis.

2️⃣ Regression Analysis – Uber Ride Price Prediction

🎯 Objective:
Predict Uber ride fares based on pickup and drop-off locations.

📂 Dataset: Uber Fares Dataset

🛠 Implementation:

Preprocessed data and handled missing values.

Detected and removed outliers.

Checked feature correlations.

Implemented Linear, Ridge, and Lasso Regression.

Evaluated models using R², RMSE, and MAE.

✅ Outcome:
Identified the best regression model for fare prediction.

3️⃣ Classification Analysis – SVM for Handwritten Digits

🎯 Objective:
Classify handwritten digits (0–9) using Support Vector Machine (SVM).

📂 Dataset: Built-in sklearn.datasets.load_digits()

🛠 Implementation:

Flattened image data into feature vectors.

Split data into training and testing sets.

Trained SVM with RBF kernel.

Evaluated accuracy using confusion matrix.

✅ Outcome:
High accuracy in digit classification, showing SVM effectiveness.

4️⃣ Clustering Analysis – K-Means on Iris Dataset

🎯 Objective:
Group similar flowers using K-Means Clustering.

📂 Dataset: Iris Dataset

🛠 Implementation:

Standardized features.

Applied K-Means.

Determined optimal cluster count using Elbow Method.

Visualized clusters with scatter plots.

✅ Outcome:
Flowers clustered effectively, matching species patterns.

5️⃣ Ensemble Learning – Random Forest for Car Safety

🎯 Objective:
Predict car safety using Random Forest Classifier.

📂 Dataset: Car Evaluation Dataset

🛠 Implementation:

Preprocessed data and encoded categorical features.

Trained Random Forest.

Evaluated with accuracy, precision, and recall.

✅ Outcome:
Accurate car safety predictions using ensemble learning.

6️⃣ Reinforcement Learning – Maze Exploration

🎯 Objective:
Implement Reinforcement Learning agent to navigate a maze.

🛠 Implementation:

Defined states, actions, and rewards.

Applied Q-Learning.

Visualized the learned path.

✅ Outcome:
Agent learned optimal path in the maze using rewards and penalties.

⚡ Prerequisites

Python 3.8+

Libraries: pandas, numpy, matplotlib, seaborn, scikit-learn

IDE: Jupyter Notebook or VS Code (with Python extension)
