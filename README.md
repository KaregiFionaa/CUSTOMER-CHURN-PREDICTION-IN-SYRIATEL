# CUSTOMER CHURN PREDICTION IN SYRIATEL
This project aims to predict customer churn for Syriatel, a telecommunications company, using historical customer data. By analyzing various factors that influence customer retention, the goal is to develop a predictive model that identifies customers likely to churn, enabling targeted retention strategies.

## 1.) Data Understanding:
Obtain a dataset from Kaggle consisting of 21 columns and 3333 rows, containing relevant information for predicting customer churn. Each row represents a customer, and each column represents features related to customer behavior, demographics, and interactions with SyriaTel's services. The columns are as follows:

* state: the state the user lives in
* account length: the number of days the user has this account
* area code: the code of the area the user lives in
* phone number: the phone number of the user
* international plan: true if the user has the international plan, otherwise false
* voice mail plan: true if the user has the voice mail plan, otherwise false
* number vmail messages: the number of voice mail messages the user has sent
* total day minutes: total number of minutes the user has been in calls during the day
* total day calls: total number of calls the user has done during the day
* total day charge: total amount of money the user was charged by the Telecom company for calls during the day
* total eve minutes: total number of minutes the user has been in calls during the evening
* total eve calls: total number of calls the user has done during the evening
* total eve charge: total amount of money the user was charged by the Telecom company for calls during the evening
* total night minutes: total number of minutes the user has been in calls during the night
* total night calls: total number of calls the user has done during the night
* total night charge: total amount of money the user was charged by the Telecom company for calls during the night
* total intl minutes: total number of minutes the user has been in international calls
* total intl calls: total number of international calls the user has done
* total intl charge: total amount of money the user was charged by the Telecom company for international calls
* customer service calls: number of customer service calls the user has done
* churn: true if the user terminated the contract, otherwise false


## 3.) Data Preparation:
Clean the dataset by handling missing values, encoding categorical variables, and scaling numerical features. Perform exploratory data analysis to understand the distribution of features and identify potential correlations with customer churn.

## 3.) Modeling:

### 3.1) Logistic Regression:
Description: Logistic Regression is a statistical method for analyzing a dataset in which there are one or more independent variables that determine an outcome.
Advantages: It's a simple and interpretable model that provides probabilities for outcomes.
Disadvantages: It assumes a linear relationship between the independent variables and the log odds of the outcome.
Implementation:used the Logistic Regression model from the scikit-learn library.


### 3.2) Decision Trees:
Description: Decision Trees are a type of supervised learning algorithm that is used for classification and regression tasks.
Advantages: They are easy to understand and visualize, and they can handle both numerical and categorical data.
Disadvantages: They are prone to overfitting, especially with complex datasets.
Implementation: utilized the DecisionTreeClassifier from the scikit-learn library.

### Model Evaluation:
Based on the performance metrics and feature importance from both the Decision Tree and Logistic Regression models, here’s the evaluation:

LR: 0.8546 DT: 0.9070 Rank: DT > LR Precision:
LR: 0.5000 DT: 0.6768 Rank: DT > LR Recall:
LR: 0.1340 DT: 0.6907 Rank: DT > LR F1 Score:
LR: 0.2114 DT: 0.6837 Rank: DT > LR ROC AUC:
LR: 0.8264 DT: 0.8173 Rank: LR > DT
● DT outperforms LR across most metrics, especially in terms of accuracy, precision,
recall, and F1 score, indicating it's more reliable at distinguishing between churn and
non-churn cases.
● LR slightly outperforms DT in ROC AUC, suggesting it might have a marginally better
ability to differentiate between the classes at different thresholds, though both models
perform similarly in this aspect.
● DT is a better overall model in this case, especially considering its higher accuracy and
ability to identify both churners and non-churners effectively

### Summary of Rankings
● DT is a better overall model in this case, especially considering its higher accuracy and
ability to identify both churners and non-churners effectively

Recommendation
Improvements: Explore ensemble methods (e.g., Random Forest, Gradient Boosting) to combine strengths of both models. Conduct further feature engineering and hyperparameter tuning to enhance predictive performance.

Focus on Churn Predictions: Address the high false positive rates and improve recall for churn predictions to enhance retention strategies effectively.

Additional Feature Engineering: Investigate additional feature engineering techniques to improve model performance.

Conclusion
Based on the evaluation metrics, Logistic Regression ranks highest due to better precision, recall, and ROC-AUC scores, indicating more reliable churn predictions. Decision Tree follows, showing strong accuracy but weaker performance in precision and recall, suggesting it struggles to accurately identify churners. Overall, Logistic Regression is preferred for effective churn prediction, while the Decision Tree offers valuable insights into feature importance.

To improve overall performance, consider exploring ensemble methods or further tuning both models. Additionally, addressing the identified weaknesses—particularly the high false positive rate in churn predictions—will enhance the effectiveness of retention efforts. Combining the strengths of both models could lead to a more robust and actionable churn prediction framework.

Next Steps
Model Selection and Implementation: Finalize the choice of Logistic Regression as the primary model for churn prediction based on its superior metrics.

Feature Engineering: Explore additional features or transformations of existing features to enhance model performance. Consider interaction terms or polynomial features if applicable.

Hyperparameter Tuning: Optimize the hyperparameters of both models using techniques like Grid Search or Random Search to improve their performance further. Ensemble Methods: Experiment with ensemble techniques (e.g., Random Forest, Gradient Boosting) to combine the strengths of both models and enhance predictive accuracy.

Model Evaluation: Implement cross-validation to ensure the robustness of the selected model. Continuously monitor model performance on new data to adapt as needed.
