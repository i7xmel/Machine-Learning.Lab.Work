# Machine Learning Lab Work

This repository contains 9 practical programs implementing fundamental machine learning techniques, from data preprocessing and exploratory analysis to classification, regression, clustering, and advanced algorithms.

## Programs Overview

### Program 1: Customer Churn Prediction - Data Preprocessing & Feature Engineering
- Analyzed Churn_Modelling dataset with 10,000 customer records
- Performed comprehensive data cleaning: removed missing values, duplicates, and irrelevant columns
- Applied Z-score normalization and outlier detection using 3-sigma rule
- Implemented Label Encoding for categorical variables and MinMaxScaler for feature scaling
- Conducted PCA dimensionality reduction with 3 principal components
- Applied KBinsDiscretizer for binning continuous variables (Balance feature)
- Generated extensive visualizations: correlation heatmaps, pairplots, boxplots, and distribution charts

**Screenshot**


<img width="406" height="357" alt="image" src="https://github.com/user-attachments/assets/ea516164-10bb-416f-b519-1382be7b7f0f" />
<img width="391" height="307" alt="image" src="https://github.com/user-attachments/assets/1364adcd-00f6-4282-ab5d-15fc57ce0535" />
<img width="425" height="352" alt="image" src="https://github.com/user-attachments/assets/2305b664-4433-461a-b720-1c7be4aa758a" />
<img width="388" height="342" alt="image" src="https://github.com/user-attachments/assets/00300de3-5c10-4637-86a0-c3e1d82a890e" />
<img width="448" height="280" alt="image" src="https://github.com/user-attachments/assets/cc726439-ef15-4c62-83e9-2dd8ef6656e1" />


---

### Program 2: Airline Customer Satisfaction Analysis
- Analyzed Invistico_Airline dataset with 129,880 customer records
- Performed feature selection by removing irrelevant columns (Inflight wifi, entertainment, etc.)
- Created age-based customer segmentation with 4 categories: Young, Middle-Aged, Older-adults, Senior-citizens
- Implemented covariance matrix analysis for categorical variables
- Conducted Chi-Square test for independence between Gender and Repeated Purchases
- Performed T-Test to analyze relationship between customer rating and repeated purchases
- Generated comprehensive visualizations including correlation matrices and bar charts

**Screenshot**


<img width="361" height="228" alt="image" src="https://github.com/user-attachments/assets/26904bfb-af65-41dc-a625-c3fc989b76e3" />
<img width="379" height="304" alt="image" src="https://github.com/user-attachments/assets/82a8d2f2-b0b3-43ee-8912-fd61ed4926da" />
<img width="357" height="310" alt="image" src="https://github.com/user-attachments/assets/2b718621-46fb-4b28-abb6-2903343517d4" />
<img width="332" height="238" alt="image" src="https://github.com/user-attachments/assets/5ef45fe1-23b1-4986-86ee-55c11fcfd3dc" />



---

### Program 3: Market Basket Analysis with Apriori Algorithm
- Implemented Association Rule Mining on Market_Basket_Optimisation dataset
- Applied Label Encoding for transaction data preprocessing
- Used Apriori algorithm with minimum support threshold of 0.06
- Generated association rules with lift metric and confidence scoring
- Identified top product combinations with highest lift and confidence values
- Created covariance heatmaps for confidence-lift relationship analysis
- Discovered optimal product bundling strategies for retail optimization

**Screenshot**

<img width="388" height="248" alt="image" src="https://github.com/user-attachments/assets/89dda842-d6ff-417c-a261-d87a3a204fc4" />
<img width="432" height="317" alt="image" src="https://github.com/user-attachments/assets/20aba568-470c-4a57-b031-0437df8532bd" />
<img width="412" height="342" alt="image" src="https://github.com/user-attachments/assets/850d6933-f516-4e4d-b972-677ac2340f97" />


---

### Program 4: Grocery Transaction Analysis - Apriori vs FP-Growth
- Analyzed Groceries_dataset with 38,765 transactions
- Implemented both Apriori and FP-Growth algorithms for frequent itemset mining
- Used TransactionEncoder for transaction data transformation
- Applied multi-threading for parallel algorithm execution and performance comparison
- Conducted time complexity analysis between Apriori and FP-Growth
- FP-Growth demonstrated superior performance with faster execution time
- Generated association rules with minimum support threshold of 0.001

**Screenshot**
<img width="448" height="245" alt="image" src="https://github.com/user-attachments/assets/49f17d6d-a73d-4a60-ae34-427ba1759808" />
<img width="429" height="324" alt="image" src="https://github.com/user-attachments/assets/2bfe9764-68f7-4b6f-9c72-d016cc62127b" />


---

### Program 5: Telco Customer Churn Prediction with Logistic Regression
- Analyzed Telco-Customer-Churn dataset with 7,043 customer records
- Performed comprehensive feature engineering including monthly_charges_fraction calculation
- Implemented Label Encoding for all categorical variables
- Built Logistic Regression model with 'saga' solver and 10,000 max iterations
- Achieved 79.2% accuracy with detailed performance metrics
- Generated ROC curves (AUC: 0.78) and Precision-Recall curves
- Created confusion matrix visualization and feature importance analysis
  

**Screenshot**

<img width="492" height="169" alt="image" src="https://github.com/user-attachments/assets/aadf5abd-b234-43fd-939d-f102e25a4799" />
<img width="470" height="350" alt="image" src="https://github.com/user-attachments/assets/e70da18e-c6c4-47db-a401-2f086070b465" />
<img width="493" height="355" alt="image" src="https://github.com/user-attachments/assets/29482fe7-7a58-4526-8bfe-9d43e1789955" />
<img width="453" height="334" alt="image" src="https://github.com/user-attachments/assets/e9c2c7cf-8912-4a33-b0a2-c2472c2f1dbf" />


---

### Program 6: Insurance Cost Prediction with Linear Regression
- Analyzed insurance dataset with 1,338 records for medical cost prediction
- Implemented One-Hot Encoding for categorical variables (region)
- Applied Label Encoding for binary categorical variables (sex, smoker)
- Built Linear Regression model for insurance charge prediction
- Achieved R² score of 0.784 with RMSE of 5796.28
- Conducted comprehensive error analysis: MAE, MSE, RSS, Explained Variance
- Generated feature importance visualization and scatter plots for actual vs predicted values

**Screenshot**

<img width="403" height="292" alt="image" src="https://github.com/user-attachments/assets/edf8a9ad-2712-4a7f-b8ae-73bac89b5c06" />
<img width="405" height="297" alt="image" src="https://github.com/user-attachments/assets/0c5e9f77-aada-4d41-b4ae-c0ab0c345dcd" />
<img width="426" height="300" alt="image" src="https://github.com/user-attachments/assets/c3ee90c8-2ccc-48fd-b708-aeda2c375ec1" />


---

### Program 7: Employee Attrition Prediction with Decision Trees
- Analyzed employee_data dataset with 14,249 records
- Performed extensive data cleaning: missing value imputation using mode
- Implemented both Gini Impurity and Entropy criteria for Decision Tree splitting
- Conducted hyperparameter tuning with min_samples_leaf and max_depth
- Achieved 96% accuracy with optimal pruning parameters
- Generated decision tree visualizations and feature importance analysis
- Created comparative analysis between Gini and Entropy splitting criteria

**Screenshot**

<img width="345" height="265" alt="image" src="https://github.com/user-attachments/assets/759bcf96-de9e-4cf7-9201-aa2e3e9eaf26" />
<img width="287" height="232" alt="image" src="https://github.com/user-attachments/assets/0b8dafa9-1721-4b9f-bee8-ce7180c6aa2f" />
<img width="423" height="259" alt="image" src="https://github.com/user-attachments/assets/300e1a5f-45cb-48a6-a55a-06b9f20022c5" />
<img width="367" height="285" alt="image" src="https://github.com/user-attachments/assets/a69bd5df-6874-4e75-858f-faeed81330d8" />
<img width="451" height="597" alt="image" src="https://github.com/user-attachments/assets/ab79be4d-3d09-4ac4-a0cc-a31892a5343c" />
<img width="423" height="289" alt="image" src="https://github.com/user-attachments/assets/031cdb61-574a-4b15-a6f0-f11f56d256ac" />


---

### Program 8: Spam Email Classification with Naive Bayes
- Implemented Multinomial Naive Bayes for spam email classification
- Used CountVectorizer for text feature extraction
- Achieved high accuracy with detailed performance metrics
- Generated comprehensive visualizations: confusion matrix, ROC curve, Precision-Recall curve
- Created feature importance analysis showing top spam indicators
- Built interactive model evaluation dashboard with multiple metrics
- Demonstrated effective text classification with probabilistic approach

**Screenshot**

<img width="413" height="329" alt="image" src="https://github.com/user-attachments/assets/ad63ba99-490a-426b-8485-7307ab016d14" />
<img width="427" height="372" alt="image" src="https://github.com/user-attachments/assets/54b86bc7-558b-4040-9bd0-91c035bcfa2b" />
<img width="417" height="364" alt="image" src="https://github.com/user-attachments/assets/bc714a05-ef1c-4ddc-a694-1297a1d66967" />
<img width="429" height="274" alt="image" src="https://github.com/user-attachments/assets/6f6bfbb1-11dd-4b2b-83e8-b7de1fc8f5db" />
<img width="445" height="323" alt="image" src="https://github.com/user-attachments/assets/1ab44ae6-3f07-4b54-aeed-702d96c2a9b5" />


---

### Program 9: Brain Tumor Detection with SVM and PCA
- Implemented medical image classification for brain tumor detection
- Used MRI dataset with 1,222 images across two classes (tumor/no tumor)
- Applied PCA for dimensionality reduction (100 components)
- Implemented SVM with linear kernel for classification
- Achieved 95.5% accuracy with comprehensive performance metrics
- Conducted hyperparameter tuning using GridSearchCV
- Generated ROC curves (AUC: 0.94) and confusion matrices
- Created patient age distribution analysis and decision boundary visualizations

**Screenshot**

<img width="610" height="457" alt="image" src="https://github.com/user-attachments/assets/3482216a-090b-4579-90de-fcdaa8b15fe7" />
<img width="583" height="374" alt="image" src="https://github.com/user-attachments/assets/677756a6-ae57-49a6-9d36-dd591650df80" />
<img width="563" height="374" alt="image" src="https://github.com/user-attachments/assets/49e7e71c-3aca-40c2-92e9-0c096cdae8d0" />
<img width="586" height="428" alt="image" src="https://github.com/user-attachments/assets/05b7ab65-ffd8-405b-865c-8908e4dbdefc" />
<img width="488" height="382" alt="image" src="https://github.com/user-attachments/assets/318c17bb-89ee-4ae5-86b2-a861b32a1ed4" />
<img width="484" height="247" alt="image" src="https://github.com/user-attachments/assets/8b639745-ccfd-4ee9-9a77-ff37d8d55f9e" />
<img width="634" height="425" alt="image" src="https://github.com/user-attachments/assets/7a9c6358-7892-45ff-ba8d-c8fe7a2f7f04" />


```
