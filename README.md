# Ticket_Customer_Support____Celebal_6

🎫 Ticket_Customer_Support - Model Evaluation & Hyperparameter Tuning





🔍 Project Objective
This project focuses on building and optimizing machine learning models to predict and classify support ticket issues based on customer queries. The aim is to evaluate different models and apply hyperparameter tuning techniques to select the most efficient model for real-world deployment.






📌 Assignment Context
Task 6 of my Data Science internship included:

Training multiple ML models on a ticket support dataset.

Evaluating performance using Accuracy, Precision, Recall, and F1-score.

Applying GridSearchCV and RandomizedSearchCV for hyperparameter tuning.

Selecting and analyzing the best-performing model with a detailed explanation of tuning parameters.






🧠 Models Implemented
Logistic Regression

Decision Tree Classifier

Random Forest Classifier

Support Vector Machine (SVM)

Gradient Boosting Classifier






📊 Evaluation Metrics
To compare model performance:

Accuracy: Overall correctness of the model.

Precision: True positives over predicted positives; useful for reducing false alarms.

Recall: True positives over actual positives; crucial in support scenarios to catch all real issues.

F1-Score: Harmonic mean of Precision and Recall; balances the two.






🛠️ Hyperparameter Tuning Techniques
1. GridSearchCV
Exhaustively tries every combination of parameters in a grid.

Used for models like SVM and Random Forest.

Example Parameters Tuned:

C, kernel in SVM

n_estimators, max_depth in Random Forest

2. RandomizedSearchCV
Samples a given number of parameter settings from specified distributions.

Faster than GridSearch for large search spaces.

Applied to Gradient Boosting for tuning learning_rate, n_estimators, subsample, etc.






📈 Best Model Summary
After tuning, Random Forest Classifier with optimized parameters gave the best performance:

Accuracy: 92%

Precision: 91%

Recall: 90%

F1-score: 90.5%






🔍 Hyperparameters:

python
Copy
Edit
{
  'n_estimators': 150,
  'max_depth': 10,
  'min_samples_split': 4,
  'min_samples_leaf': 2
}





📂 Project Structure
kotlin
Copy
Edit
Ticket_Customer_Support/
│
├── data/
│   └── ticket_dataset.csv
├── notebooks/
│   └── model_evaluation.ipynb
├── results/
│   ├── model_comparison.png
│   └── best_model_metrics.json
├── ticket_model.py
├── requirements.txt
└── README.md





✅ Key Learnings
Importance of evaluation metrics beyond accuracy.

Hands-on experience with GridSearchCV and RandomizedSearchCV.

Parameter tuning can significantly improve model generalization.

Effective model selection requires both performance metrics and practical considerations (speed, interpretability, etc.).






📌 Tools & Libraries
scikit-learn

pandas

matplotlib, seaborn

joblib for model persistence

📢 Conclusion
This project solidified my understanding of how model evaluation and tuning play a critical role in building reliable AI systems, especially in customer-centric applications like support ticket classification.

