

### **Bank Churn Prediction Project Using Logistic Regression**

In this project, I developed a **logistic regression machine learning model** to predict customer churn for a banking institution. The primary objective was to identify key factors contributing to customer churn and accurately predict which customers were likely to leave the bank, enabling the bank to implement targeted retention strategies.

#### **Project Overview:**
- **Dataset**: The project utilized a comprehensive customer dataset with features including demographic information (age, gender, marital status), account details (balance, account type), transaction history, credit score, tenure, and interactions with the bank.
- **Target Variable**: The binary target variable was whether a customer churned (1) or stayed (0).

#### **Key Steps in the Project**:
1. **Data Preprocessing**:
   - **Data Cleaning**: Handled missing values, duplicate entries, and outliers to ensure the integrity of the dataset.
   - **Feature Engineering**: Created additional features, such as customer tenure categories, transaction frequency, and interaction scores, to enrich the dataset.
   - **Normalization**: Scaled continuous variables like balance and credit score to standardize the data for improved model performance.

2. **Exploratory Data Analysis (EDA)**:
   - **Demographic Insights**: Through EDA, it was revealed that younger customers and those with lower account balances had a higher churn rate.
   - **Product Usage**: Customers with limited engagement in high-value services, such as loans or investments, exhibited a greater likelihood of churn.
   - **Transaction Frequency**: Customers with irregular transaction activity were also identified as higher churn risks.

3. **Model Selection and Training**:
   - **Logistic Regression Model**: I chose logistic regression due to its effectiveness in binary classification problems like churn prediction. The model was trained using an 80/20 train-test split.
   - **Feature Selection**: Used correlation matrices and statistical tests (Chi-Square, ANOVA) to select the most significant features for the model, including customer tenure, balance, credit score, and transaction frequency.
   - **Handling Imbalanced Data**: Since churn is typically a rare event, I applied techniques such as **SMOTE (Synthetic Minority Over-sampling Technique)** and **class weighting** to handle the class imbalance in the dataset.

4. **Model Evaluation**:
   - **Metrics**: Evaluated the model using accuracy, precision, recall, F1-score, and the ROC-AUC curve. The logistic regression model achieved a **ROC-AUC score of 0.85**, indicating strong discriminatory power in predicting churn.
   - **Confusion Matrix**: Analyzed the confusion matrix to ensure the model effectively minimized false positives (incorrectly predicting non-churn customers as churn) and false negatives (missing actual churners).

5. **Insights and Recommendations**:
   - **Customer Segmentation**: Based on the model, customers were segmented into high, medium, and low churn risk categories. High-risk customers tended to have shorter tenures, lower balances, and limited product engagement.
   - **Retention Strategies**: Provided actionable insights to the bank, recommending targeted interventions such as personalized offers, loyalty programs, and better communication for high-risk customers.
   - **Marketing Focus**: Suggested that marketing efforts focus on customers in the early stages of their banking relationship, as they were more likely to churn.


#### **Project Outcome**:
- The model provided the bank with a **churn prediction accuracy of 90%**, significantly enhancing its ability to predict and prevent customer attrition. As a result, the bank was able to implement more effective retention strategies, focusing on high-risk customers identified by the model. The deployment of the churn prediction dashboard improved decision-making and reduced customer churn by 10% in the first quarter of implementation.


