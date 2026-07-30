# baby-ml
### notes on SciKit-Learn to understand how to create my own ML model

## Skit-Learn 

Scikit learn is an open-source python library for machine learning, it provides tools for data analysis, and predictive modeling through offering simple and reusable functions to build models for tasks such as classification regression, clustering, dimensionality reduction, feature extraction and cross validation.

## Key Features of Scikit are: 

-DATA PROCESSING --> 
a. Data splitting - Divide data into training and testing sets
b. Feature scaling - Normalize or standardize features values
c. Feature selection - Choose most relevant features
d. Feature extraction - Create new features from existing data 

-MODEL EVALUATION 
a. Metrics - Evaluate model performance for accuracy, precision, recall
b. Model Selection - Tools for selecting the best model hyper parameters through techniques like grid search + randomized search

-PIPELINE SUPPORT --> Preprocessing + modeling 
-INTEGRATION --> Works with existing python libraries 

__________________________________________________________________________________________________________________________________
## Machine Learning Techniques Supported by Scikit Learn:

### SUPERVISED - 
Training models using labeled data, where the correct output is already known.
-Classification (categorical) - algorithms to predict categorical outcomes, such as logistic regression, decision trees, random forests, support vector machines (SVMs) and gradient boosting.
-Regression (numerical) - predict continuous numerical values. Scikit-learn supports linear regression, support vector regression and decision tree regression.


### UNSUPORVISED - 
Learning through unlabeled data to discover patterns and structure.
-Clustering - techniques to group similar data points, including K-means clustering, DBSCAN and hierarchical clustering.
-Dimensionality Reduction - handle high-dimensional data efficiently, Scikit-learn provides techniques like principal component analysis (PCA).
_____________________________________________________________________________________________________________________________
## Building Learning Model w/ Scikit Learn - pip install -U scikit-learn 

1. LOADING A DATASET
Dataset = Features(X) Input variables that describes the data + Target(y) the value we want to predict
* To create custom data use Python PANDAS
  
2. SPLITTING THE DATASET
To evaluate the model fairly, split data into the Training set (used to train the model) and Testing set(used to evaluate how well the model generalizes)

3. HANDLING CATEGORICAL DATA
ML algorithms work with numerical inputs so categorical (text data must be converted into numbers. If not encoded properly, models can misinterpret categories.
*Label Encoding -
LabelEncoder(): create an encoder object that will convert categorical values into numerical labels.
fit_transform():  fits the encoder to the categorical data and then transforms the categories into corresponding numeric labels.
*One Hot Encoding -creates separate binary columns for each category. This is useful when categories do not have any natural ordering. Input must be reshaped into a 2D array, and OneHotEncoder(sparse_output=False) generates binary columns

4. TRANING THE MODEL - Scikit-learn has many algorithms with a consistent interface for training, prediction and evaluation. (training and predictions vary for algorithm type)

5. MAKE PREDICTIONS - Once trained we use the model to make predictions on the test data X_test by calling the predict method. These returns predicted labels y_pred.

6. EVALUATE MODEL ACCURACY - Check how well our model is performing by comparing y_test and y_pred. Now we want our model to make predictions on new sample data. Then the sample input can simply be passed in the same way as we pass any feature matrix.

__________________________________________________________________________________________________________________________________
## Data Normalization - 

Helps to mitigate this issue by transforming features to a common scale, ensuring that all features contribute equally to the model.

## Transformers for Normalization: 

### MinMaxScaler - 
Min-max normalization, also known as feature scaling, is a widely used technique that rescales features to a common range, typically between 0 and 1. This technique is useful when the ranges of features vary significantly.

- Initialize the scaler
scaler = MinMaxScaler()

- Fit and transform the data
normalized_data = scaler.fit_transform(data)
print("Normalized Data (Min-Max Scaling):")
print(normalized_data)

### StandardScaler - 
Z-score normalization, also known as standardization, transforms features to follow a standard normal distribution with a mean of 0 and a standard deviation of 1. This technique is

- Initialize the scaler
scaler = StandardScaler()

- Fit and transform the data
standardized_data = scaler.fit_transform(data)
print("Standardized Data (Z-score Normalization):")
print(standardized_data)

### RobustScaler -  
Robust Scaling uses the median and the interquartile range to scale features, making it robust to outliers

- Initialize the scaler
scaler = RobustScaler()

- Fit and transform the data
robust_scaled_data = scaler.fit_transform(data)
print("Robust Scaled Data:")
print(robust_scaled_data)

__________________________________________________________________________________________________________________________________
## Data Preprocessing - 

Cleans and transformers raw data into a format suitable for modeling. ColumnTransformer is a powerful tool that allows you to apply different transformations to different subsets of features within your dataset. 

### ColumnTransfer - 
allows you to selectively apply data preparation transforms to different columns in your dataset. This is particularly useful when you have a mix of numerical and categorical data that require different preprocessing steps.

from sklearn.compose import ColumnTransformer

transformer = ColumnTransformer(transformers=[('imputer', SimpleImputer(), ['NumericalColumn1', 'NumericalColumn2']),('ordinal', OrdinalEncoder(), ['OrdinalColumn']),('onehot', OneHotEncoder(), ['CategoricalColumn1', 'CategoricalColumn2'])],remainder='passthrough')


*Selective Transformation - apply specific transformations to subsets of columns 
*Pipeline Integration - Integrate with Pipeline for streamlined workflows 
*Code Organization - encapsulates preprocessing logic into single maintainable object

## Implementing ColumnTransfer 
1.Import necessary libraries (SimpleImputer, OneHotEncoder, OrdinalEncoder)
2. Load the Dataset
3. Concatenate into single transformer ( using the np.concatenate function, we ensure that each feature retains its processed form and contributes equally to the predictive modeling task.)

__________________________________________________________________________________________________________________________________
## Feature Selection - 

Process of identifying and selecting a subset of relevant features for use in model construction. The goal is to enhance the model's performance by reducing overfitting, improving accuracy, and reducing training time.

*Improved Model Preformance
*Reduced Overfitting 
*Faster

### Types of Feature Selection Methods: 

- FILTER METHODS -  use statistical techniques to evaluate the relevance of features independently of the model. Common techniques include correlation coefficients, chi-square tests, and mutual information.
- WRAPPER METHODS - use a predictive model to evaluate feature subsets and select the best-performing combination. Techniques include recursive feature elimination (RFE) and forward/backward feature selection.
- EMBEDDED METHODS - perform feature selection during the model training process. Examples include Lasso (L1 regularization) and feature importance from tree-based models.

### Feature Selection Techniques: 
- UNIVARIATE SELECTION - evaluates each feature individually to determine its importance. Techniques like SelectKBest and SelectPercentile can be used to select the top features based on statistical tests.
- RECURSIVE FEATURE ELIMINATION (RFE) - wrapper method that recursively removes the least important features based on a model's performance. It repeatedly builds a model and eliminates the weakest features until the desired number of features is reached.
- FEATURE IMPORTANCE FROM TREE BASED MODELS -  Tree-based models like decision trees and random forests can provide feature importance scores, indicating the importance of each feature in making predictions.

__________________________________________________________________________________________________________________________________
## Pipelines For Structured Workflows 

The Pipeline class in scikit-learn is a powerful tool designed to streamline the machine learning workflow. It allows you to chain together multiple steps, such as data transformations and model training, into a single, cohesive process. This simplifies the code but also ensures that the same sequence of steps is applied consistently to both training and testing data, thereby reducing the risk of data leakage and improving reproducibility.

*Code Readability and Maintaining 
*Reproducibility
*HyperParamter Tuning (GridSearchCV) + (RandomizedSearchCV)
*Modularity

## Components of a PipeLine 

A pipeline in scikit-learn consists of a sequence of steps, where each step is a tuple containing a name and a transformer or estimator object.
The final step in the pipeline must be an estimator (e.g., a classifier or regressor), while the preceding steps must be transformers (e.g., scalers, encoders).

Ex. 

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=2)),
    ('classifier', LogisticRegression())
])

StandardScaler: Scales the features to have zero mean and unit variance.
PCA: Reduces the dimensionality of the data to two principal components.
LogisticRegression: Trains a logistic regression model on the transformed data.

## Creating ML Pipeline for Scikit 

1. Import the necessary libraries and load your dataset
2. Define the pipeline by specifying the sequence of steps.
3. Fit the pipeline on the training data : pipeline.fit(X_train, y_train)
4. Use the trained pipeline to make predictions on the test data: y_pred = pipeline.predict(X_test)



