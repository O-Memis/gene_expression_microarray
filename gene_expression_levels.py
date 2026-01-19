"""
Classification of Gene Expression Levels from a benchmark Microarray dataset

Oguzhan Memis             22/01/2026


Dataset link: https://csse.szu.edu.cn/staff/zhuzx/Datasets.html

Original paper: "Markov blanket-embedded genetic algorithm for gene selection" from Zhu et. al. 2007
"""


#%% 1) Importing datasets


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
from scipy.io import arff




# 1) SRBCT DATASET
# arff module of scipy works for SRBCT data (no duplicate columns)
data, meta = arff.loadarff('SRBCT.arff')
df_srbct = pd.DataFrame(data)

# Decode class labels from bytes to integers
class_col = df_srbct.columns[-1]
df_srbct[class_col] = df_srbct[class_col].str.decode('utf-8')  # (corrected from b'1' to 1)
df_srbct[class_col] = df_srbct[class_col].astype(int)          # converted into integer




# 2) COLON DATASET 
# Manual parsing needed (Colon has duplicate gene names like 'R39465')

with open('Colon.arff', 'r') as f:
    
    lines = f.readlines()


# Find where actual data starts (after @DATA line)
data_start = next(i for i, line in enumerate(lines) if line.strip().upper() == '@DATA')


# Extract column names from @ATTRIBUTE lines
columns = []
for line in lines[:data_start]:
    
    if line.strip().upper().startswith('@ATTRIBUTE'):
        
        col_name = line.split()[1]  # Second word is column name
        columns.append(col_name)


# Handle duplicate column names by adding suffix (_1, _2, etc.)
seen = {}
unique_cols = []

for col in columns:
    
    if col in seen:
        seen[col] += 1
        unique_cols.append(f"{col}_{seen[col]}")  # R39465_1, R39465_2
    else:
        seen[col] = 0
        unique_cols.append(col)


# Parse data rows (skip empty lines)
data_lines = [line.strip() for line in lines[data_start+1:] if line.strip()]
data_values = [line.split(',') for line in data_lines]

df_colon = pd.DataFrame(data_values, columns=unique_cols)


# Convert gene expression values to numeric (last column is class label)
for col in df_colon.columns[:-1]:
    
    df_colon[col] = pd.to_numeric(df_colon[col], errors='coerce')




print(f"\nSRBCT shape: {df_srbct.shape}")
print(f"Colon shape: {df_colon.shape}")
print(f"\nSRBCT classes: {df_srbct.iloc[:, -1].unique()}")
print(f"Colon classes: {df_colon.iloc[:, -1].unique()}")
print(f"\nSRBCT class counts:\n{df_srbct.iloc[:, -1].value_counts()}")
print(f"\nColon class counts:\n{df_colon.iloc[:, -1].value_counts()}")



# For easier analysis, we can encode the string labels into Tumor=1 and Normal=0
df_colon.iloc[:, -1] = df_colon.iloc[:, -1].map({'Tumor': 1, 'Normal': 0})
df_colon['class'] = df_colon['class'].astype(int)




#%% 2) EDA 


# 1) EDA for SRBCT Dataset

print("\n ---SRBCT Dataset--- \n ")

print(df_srbct.describe())                 # Basic statistics
print(df_srbct.info())                     # Data types
print(df_srbct[class_col].value_counts())  # Class distribution


# 1.1 Plotting class distribution
plt.figure(figsize=(15, 10))
sns.countplot(x=class_col, data=df_srbct, palette='Set2', hue=class_col, legend=False)
plt.title('SRBCT Class Distribution')
plt.xlabel('Cancer Type')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()


# 1.2 Plotting boxplots for gene expression of the first 20 selected genes
selected_genes = df_srbct.columns[:-1][:20]  
plt.figure(figsize=(20, 15))  # Adjusted size for more subplots
for i, gene in enumerate(selected_genes):
    plt.subplot(5, 4, i + 1)  # 5 rows and 4 columns for 20 genes
    sns.boxplot(x=class_col, y=gene, data=df_srbct, palette='Set2', hue=class_col, legend=False)
    plt.title(f'Expression of {gene} by Class')
plt.tight_layout()
plt.show()



# 1.3 Summary statistics

stats0 = df_srbct[selected_genes].describe()
stats0 = stats0.drop(index='count') # the count information is redundant after now

plt.figure(figsize=(20, 15))

sns.heatmap(stats0, annot=True, cmap='Purples', cbar=True)
plt.title('Summary statistics of SRBCT Data')
plt.xlabel('Statistics')
plt.ylabel('Features')



# 1.4 Plotting histograms for gene expression distribution
plt.figure(figsize=(20, 15))  # Adjusted size for more subplots
for i, gene in enumerate(selected_genes):
    plt.subplot(5, 4, i + 1)  # 5 rows and 4 columns for 20 genes
    sns.histplot(df_srbct[gene], bins=30, kde=True, color='blue')
    plt.title(f'Histogram of {gene} Expression')
plt.tight_layout()
plt.show()


# 1.5 Correlation heatmap for the first few genes
correlation_matrix = df_srbct[selected_genes].corr()
plt.figure(figsize=(15, 10))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='Purples', square=True, vmin=-0.5, vmax=1)
plt.title('Correlation Heatmap for Selected Genes (SRBCT)')
plt.show()


# 1.6 Plotting violin plots for gene expression of the first 20 selected genes
plt.figure(figsize=(20, 15))  # Adjusted size for more subplots
for i, gene in enumerate(selected_genes):
    plt.subplot(5, 4, i + 1)  # 5 rows and 4 columns for 20 genes
    sns.violinplot(x=class_col, y=gene, data=df_srbct, palette='Set2', hue=class_col, legend=False)
    plt.title(f'Violin Plot of {gene} by Class')
plt.tight_layout()
plt.show()




# 2) EDA for Colon Dataset

print("\n ---Colon Dataset--- \n ")

print(df_colon.describe())                 # Basic statistics
print(df_colon.info())                     # Data types
print(df_colon['class'].value_counts())    # Class distribution

# 2.1 Plotting class distribution
plt.figure(figsize=(15, 10))
sns.countplot(x='class', data=df_colon, palette='Set2', hue='class', legend=False)
plt.title('Colon Class Distribution')
plt.xlabel('Tumor (1) vs Normal (0)')
plt.ylabel('Count')
plt.xticks(ticks=[0, 1], labels=['Normal', 'Tumor'])
plt.show()


# 2.2 Plotting boxplots for gene expression of the first 20 selected genes
selected_genes_colon = df_colon.columns[:-1][:20]  
plt.figure(figsize=(20, 15))  # Adjusted size for more subplots
for i, gene in enumerate(selected_genes_colon):
    plt.subplot(5, 4, i + 1)  # 5 rows and 4 columns for 20 genes
    sns.boxplot(x='class', y=gene, data=df_colon, palette='Set2', hue='class', legend=False)
    plt.title(f'Expression of {gene} by Class')
plt.tight_layout()
plt.show()



# 2.3 Summary statistics

stats1 = df_colon[selected_genes_colon].describe()
stats1 = stats1.drop(index='count') 

plt.figure(figsize=(21, 15))

sns.heatmap(stats1, annot=True, cmap='Purples', cbar=True, fmt=".2f")
plt.title('Summary statistics of Colon Data')
plt.xlabel('Statistics')
plt.ylabel('Features')



# 2.4 Plotting histograms for gene expression distribution
plt.figure(figsize=(20, 15))  # Adjusted size for more subplots
for i, gene in enumerate(selected_genes_colon):
    plt.subplot(5, 4, i + 1)  # 5 rows and 4 columns for 20 genes
    sns.histplot(df_colon[gene], bins=30, kde=True, color='green')
    plt.title(f'Histogram of {gene} Expression')
plt.tight_layout()
plt.show()


# 2.5 Correlation heatmap for the first few genes
correlation_matrix_colon = df_colon[selected_genes_colon].corr()
plt.figure(figsize=(15, 10))
sns.heatmap(correlation_matrix_colon, annot=True, fmt=".2f", cmap='Purples', square=True, vmin=-0.5, vmax=1)
plt.title('Correlation Heatmap for Selected Genes (Colon)')
plt.show()


# 2.6 Plotting violin plots for gene expression of the first 20 selected genes
plt.figure(figsize=(20, 15))  # Adjusted size for more subplots
for i, gene in enumerate(selected_genes_colon):
    plt.subplot(5, 4, i + 1)  # 5 rows and 4 columns for 20 genes
    sns.violinplot(x='class', y=gene, data=df_colon, palette='Set2', hue='class', legend=False)
    plt.title(f'Violin Plot of {gene} by Class')
plt.tight_layout()
plt.show()




#%% 3) Train-test split

"""
To prevent data leakage (inclusion of effect of the test data, into the analytics-based operations)
we should separate the training and testing (UNSEEN) sets before the selection & normalization.
"""



from sklearn.model_selection import train_test_split


# Splitting SRBCT dataset into training and test sets (80-20 ratio)
X_srbct = df_srbct.iloc[:, :-1]    # Features
y_srbct = df_srbct[class_col]      # Target variable
y_srbct = df_srbct[class_col] - 1  # Target variable remapped to 0, 1, 2, 3
X_train_srbct, X_test_srbct, y_train_srbct, y_test_srbct = train_test_split(X_srbct, y_srbct, test_size=0.2, random_state=42)



# Splitting Colon dataset into training and test sets (80-20 ratio)
X_colon = df_colon.iloc[:, :-1]    # Features
y_colon = df_colon['class']        # Target variable
X_train_colon, X_test_colon, y_train_colon, y_test_colon = train_test_split(X_colon, y_colon, test_size=0.2, random_state=42)


print(f"SRBCT Training set shape: {X_train_srbct.shape}")
print(f"SRBCT Test set shape: {X_test_srbct.shape}")
print(f"Colon Training set shape: {X_train_colon.shape}")
print(f"Colon Test set shape: {X_test_colon.shape}")



#%% 4.a) Feature Selection Option 1: Mutual Information



from sklearn.feature_selection import mutual_info_classif
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



# 1) Mutual Information for SRBCT Training Set
mutual_info_srbct = mutual_info_classif(X_train_srbct, y_train_srbct)
mutual_info_df_srbct = pd.DataFrame(mutual_info_srbct, index=X_train_srbct.columns, columns=['Mutual Information'])
mutual_info_df_srbct = mutual_info_df_srbct.sort_values(by='Mutual Information', ascending=False)


# Select the top 50 features based on Mutual Information
top_features_srbct = mutual_info_df_srbct.head(50).index
print("\n--- Top 50 Features by Mutual Information (SRBCT) ---\n")
print(top_features_srbct)


# Plotting the mutual information scores for SRBCT
plt.figure(figsize=(20, 15))
sns.barplot(x=mutual_info_df_srbct.index[:50], y='Mutual Information', data=mutual_info_df_srbct.head(50), palette='Set2')
plt.title('Top 50 Mutual Information Scores for Features (SRBCT)')
plt.xlabel('Features')
plt.ylabel('Mutual Information Score')
plt.xticks(rotation=45)
plt.show()




# 2) Mutual Information for Colon Training Set
mutual_info_colon = mutual_info_classif(X_train_colon, y_train_colon)
mutual_info_df_colon = pd.DataFrame(mutual_info_colon, index=X_train_colon.columns, columns=['Mutual Information'])
mutual_info_df_colon = mutual_info_df_colon.sort_values(by='Mutual Information', ascending=False)


# Select the top 50 features based on Mutual Information
top_features_colon = mutual_info_df_colon.head(50).index
print("\n--- Top 50 Features by Mutual Information (Colon) ---\n")
print(top_features_colon)


# Plotting the mutual information scores for Colon
plt.figure(figsize=(20, 15))
sns.barplot(x=mutual_info_df_colon.index[:50], y='Mutual Information', data=mutual_info_df_colon.head(50), palette='Set2')
plt.title('Top 50 Mutual Information Scores for Features (Colon)')
plt.xlabel('Features')
plt.ylabel('Mutual Information Score')
plt.xticks(rotation=45)
plt.show()



# Transforming the training sets to include only the selected features
X_train_srbct_selected = X_train_srbct[top_features_srbct]
X_train_colon_selected = X_train_colon[top_features_colon]

# Then apply this selection to the test sets
X_test_srbct_selected = X_test_srbct[top_features_srbct]
X_test_colon_selected = X_test_colon[top_features_colon]

print(f"Transformed SRBCT Training set shape: {X_train_srbct_selected.shape}")
print(f"Transformed SRBCT Test set shape: {X_test_srbct_selected.shape}")
print(f"Transformed Colon Training set shape: {X_train_colon_selected.shape}")
print(f"Transformed Colon Test set shape: {X_test_colon_selected.shape}")




#%% 4.b) Feature Selection Option 2: Recursive Feature Elimination (RFE)



from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns


# 1) RFE for SRBCT Training Set
model_srbct = LogisticRegression(max_iter=2000)
rfe_srbct = RFE(estimator=model_srbct, n_features_to_select=50)
rfe_srbct.fit(X_train_srbct, y_train_srbct)


# Get the selected features for SRBCT
selected_features_rfe_srbct = X_train_srbct.columns[rfe_srbct.support_]
print("\n--- Selected Features by RFE (SRBCT) ---\n")
print(selected_features_rfe_srbct)


# Plotting the ranking of selected features for SRBCT
plt.figure(figsize=(20, 15))
sns.barplot(x=selected_features_rfe_srbct, y=rfe_srbct.ranking_[rfe_srbct.support_], palette='Set2')
plt.title('Feature Ranking by RFE (SRBCT)')
plt.xlabel('Features')
plt.ylabel('Ranking')        # All selected features are assigned to rank=1 , so this plot is not very informative
plt.xticks(rotation=45)
plt.show()



# Extract feature importance from the fitted estimator
feature_importance_srbct = np.abs(rfe_srbct.estimator_.coef_).mean(axis=0)

# Create DataFrame for sorting
rfe_importance_df_srbct = pd.DataFrame({
    'Feature': selected_features_rfe_srbct,
    'Importance': feature_importance_srbct
}).sort_values(by='Importance', ascending=False)


# Plotting the sorted feature importances
plt.figure(figsize=(20, 15))
sns.barplot(x='Feature', y='Importance', data=rfe_importance_df_srbct, palette='Set2')
plt.title('Top 50 Features by RFE - Sorted by Importance (SRBCT)')
plt.xlabel('Features')
plt.ylabel('Feature Importance (Mean Absolute Coefficient)')
plt.xticks(rotation=45)
plt.show()




# 2) RFE for Colon Training Set
model_colon = LogisticRegression(max_iter=2000)
rfe_colon = RFE(estimator=model_colon, n_features_to_select=50)
rfe_colon.fit(X_train_colon, y_train_colon)


# Get the selected features for Colon
selected_features_rfe_colon = X_train_colon.columns[rfe_colon.support_]
print("\n--- Selected Features by RFE (Colon) ---\n")
print(selected_features_rfe_colon)


# Plotting the ranking of selected features for Colon
plt.figure(figsize=(20, 15))
sns.barplot(x=selected_features_rfe_colon, y=rfe_colon.ranking_[rfe_colon.support_], palette='Set2')
plt.title('Feature Ranking by RFE (Colon)')
plt.xlabel('Features')
plt.ylabel('Ranking')        # All selected features = rank 1 
plt.xticks(rotation=45)
plt.show()



# Extract feature importance from the fitted estimator
feature_importance_colon = np.abs(rfe_colon.estimator_.coef_).flatten()  

# Create DataFrame for sorting
rfe_importance_df_colon = pd.DataFrame({
    'Feature': selected_features_rfe_colon,
    'Importance': feature_importance_colon
}).sort_values(by='Importance', ascending=False)


# Plotting the sorted feature importances 
plt.figure(figsize=(20, 15))
sns.barplot(x='Feature', y='Importance', data=rfe_importance_df_colon, palette='Set2')
plt.title('Top 50 Features by RFE - Sorted by Importance (Colon)')
plt.xlabel('Features')
plt.ylabel('Feature Importance (Absolute Coefficient)')
plt.xticks(rotation=45)
plt.show()




# Transforming the training sets to include only the selected features from RFE
X_train_srbct_selected = X_train_srbct[selected_features_rfe_srbct]
X_train_colon_selected = X_train_colon[selected_features_rfe_colon]

# Then apply this selection to the test sets
X_test_srbct_selected = X_test_srbct[selected_features_rfe_srbct]
X_test_colon_selected = X_test_colon[selected_features_rfe_colon]

print(f"Transformed SRBCT Training set shape (RFE): {X_train_srbct_selected.shape}")
print(f"Transformed SRBCT Test set shape (RFE): {X_test_srbct_selected.shape}")
print(f"Transformed Colon Training set shape (RFE): {X_train_colon_selected.shape}")
print(f"Transformed Colon Test set shape (RFE): {X_test_colon_selected.shape}")



#%% 4.c) Feature Selection Option 3: Lasso Regression


from sklearn.linear_model import LassoCV



# 1) Lasso Regression for SRBCT Training Set
lasso_srbct = LassoCV(alphas=np.logspace(-4, 4, 100), cv=5, random_state=42)
lasso_srbct.fit(X_train_srbct, y_train_srbct)


# Get the coefficients and selected features for SRBCT
lasso_coef_srbct = lasso_srbct.coef_

# Select features based on non-zero coefficients
selected_features_lasso_srbct = X_train_srbct.columns[lasso_coef_srbct != 0]


# Sort features by their absolute coefficient value
sorted_indices_srbct = np.argsort(np.abs(lasso_coef_srbct))[::-1]
top_indices_srbct = sorted_indices_srbct[:50]  # Get top 50 features
top_features_lasso_srbct = X_train_srbct.columns[top_indices_srbct]
top_coef_lasso_srbct = lasso_coef_srbct[top_indices_srbct]


print("\n--- Selected Features by Lasso Regression (SRBCT) ---\n")
print(top_features_lasso_srbct)


# Plotting the actual coefficients of the selected Lasso features for SRBCT
plt.figure(figsize=(20, 15))
sns.barplot(x=top_features_lasso_srbct, y=top_coef_lasso_srbct, palette='Set2')  
plt.title('Top 50 Lasso Coefficients (SRBCT)')
plt.xlabel('Features')
plt.ylabel('Coefficient Value')
plt.xticks(rotation=45)
plt.axhline(0, color='black', linestyle='--')
plt.show()


# Sort these coefficient values by absolute value
plt.figure(figsize=(20, 15))
sns.barplot(x=top_features_lasso_srbct, y=np.abs(top_coef_lasso_srbct), palette='Set2')
plt.title('Top 50 Lasso Features Sorted by Absolute Coefficient (SRBCT)')
plt.xlabel('Features')
plt.ylabel('Absolute Coefficient Value')
plt.xticks(rotation=45)
plt.show()





# 2) Lasso Regression for Colon Training Set
lasso_colon = LassoCV(alphas=np.logspace(-4, 4, 100), cv=5, random_state=42)
lasso_colon.fit(X_train_colon, y_train_colon)


# Get the coefficients and selected features for Colon
lasso_coef_colon = lasso_colon.coef_

# Select features based on non-zero coefficients
selected_features_lasso_colon = X_train_colon.columns[lasso_coef_colon != 0]


# Sort features by their absolute coefficient value
sorted_indices_colon = np.argsort(np.abs(lasso_coef_colon))[::-1]
top_indices_colon = sorted_indices_colon[:50]  # Get top 50 features
top_features_lasso_colon = X_train_colon.columns[top_indices_colon]
top_coef_lasso_colon = lasso_coef_colon[top_indices_colon]


print("\n--- Selected Features by Lasso Regression (Colon) ---\n")
print(top_features_lasso_colon)


# Plotting the actual coefficients of the selected Lasso features for Colon
plt.figure(figsize=(20, 15))
sns.barplot(x=top_features_lasso_colon, y=top_coef_lasso_colon, palette='Set2')
plt.title('Top 50 Lasso Coefficients (Colon)')
plt.xlabel('Features')
plt.ylabel('Coefficient Value')
plt.xticks(rotation=45)
plt.axhline(0, color='black', linestyle='--')
plt.show()


# Sort these coefficient values by absolute value
plt.figure(figsize=(20, 15))
sns.barplot(x=top_features_lasso_colon, y=np.abs(top_coef_lasso_colon), palette='Set2')
plt.title('Top 50 Lasso Features Sorted by Absolute Coefficient (Colon)')
plt.xlabel('Features')
plt.ylabel('Absolute Coefficient Value')
plt.xticks(rotation=45)
plt.show()




# Transforming the training sets to include only the selected features from Lasso
X_train_srbct_selected = X_train_srbct[top_features_lasso_srbct]
X_train_colon_selected = X_train_colon[top_features_lasso_colon]

# Then apply this selection to the test sets
X_test_srbct_selected = X_test_srbct[top_features_lasso_srbct]
X_test_colon_selected = X_test_colon[top_features_lasso_colon]

print(f"Transformed SRBCT Training set shape (Lasso): {X_train_srbct_selected.shape}")
print(f"Transformed SRBCT Test set shape (Lasso): {X_test_srbct_selected.shape}")
print(f"Transformed Colon Training set shape (Lasso): {X_train_colon_selected.shape}")
print(f"Transformed Colon Test set shape (Lasso): {X_test_colon_selected.shape}")




#%% 5) Normalization


"""
Run one of the 3 feature selection options, to compare their effect.
"""


# From dataframe to Numpy arrays for simplicity
x_train_srbct_selected = X_train_srbct_selected.to_numpy()
x_train_colon_selected = X_train_colon_selected.to_numpy()


x_test_srbct_selected = X_test_srbct_selected.to_numpy()
x_test_colon_selected = X_test_colon_selected.to_numpy()




from sklearn.preprocessing import StandardScaler



# 1) Standard normalization for SRBCT-Selected

scaler1 = StandardScaler()  

scaler1.fit(x_train_srbct_selected)   # construct the function 


srbct_train_normalized = scaler1.transform(x_train_srbct_selected)
srbct_test_normalized = scaler1.transform(x_test_srbct_selected)



# 2) Standard normalization for Colon-Selected

scaler2 = StandardScaler()  

scaler2.fit(x_train_colon_selected)   


colon_train_normalized = scaler2.transform(x_train_colon_selected)
colon_test_normalized = scaler2.transform(x_test_colon_selected)


"""
In the real-life deployment case, you won't have the test statistics. 
To simulate this scenario, we can use traiaing statistics to normalize the test data.
"""


#%% 6.1.1) Classifier with CV 1: RF (SRBCT)
    

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix



# Define the parameter grid for Random Forest
rf_param_grid_srbct = {
    'n_estimators': [50, 100, 200],
    'max_features': ['sqrt', 'log2'],
    'min_samples_split': [2, 5, 10],
    'max_depth': [None, 10, 20, 30]
}


rf_model_srbct = RandomForestClassifier(random_state=42)

rf_grid_search_srbct = GridSearchCV(rf_model_srbct, rf_param_grid_srbct, cv=5, n_jobs=-1, verbose=0, scoring='accuracy')

rf_grid_search_srbct.fit(srbct_train_normalized, y_train_srbct)


# Get the best parameters and score
best_rf_params_srbct = rf_grid_search_srbct.best_params_
best_rf_score_srbct = rf_grid_search_srbct.best_score_

print(f"Best Parameters (RF SRBCT): {best_rf_params_srbct}")
print(f"Best Cross-Validation Score (RF SRBCT): {best_rf_score_srbct}")



# Predictions
best_rf_model_srbct = rf_grid_search_srbct.best_estimator_
y_pred_rf_srbct = best_rf_model_srbct.predict(srbct_test_normalized)


# Calculate metrics
accuracy_rf_srbct = accuracy_score(y_test_srbct, y_pred_rf_srbct)
f1_rf_srbct = f1_score(y_test_srbct, y_pred_rf_srbct, average='micro')
recall_rf_srbct = recall_score(y_test_srbct, y_pred_rf_srbct, average='micro')
precision_rf_srbct = precision_score(y_test_srbct, y_pred_rf_srbct, average='micro')


print("\nTest Data Metrics (RF SRBCT)")
print(f"Test Accuracy: {accuracy_rf_srbct * 100:.2f}%")
print(f"Test F1 Score: {f1_rf_srbct * 100:.2f}%")
print(f"Test Recall: {recall_rf_srbct * 100:.2f}%")
print(f"Test Precision: {precision_rf_srbct * 100:.2f}%")



# Confusion Matrix Visualization
cm_rf_srbct = confusion_matrix(y_test_srbct, y_pred_rf_srbct)
print("\nConfusion Matrix (RF SRBCT):")
print(cm_rf_srbct)

plt.figure(figsize=(10, 8))
sns.heatmap(cm_rf_srbct, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.title('Confusion Matrix - RF SRBCT')
plt.show()


#%% 6.1.2) Classifier with CV 1: RF (Colon)


# Define the parameter grid for Random Forest
rf_param_grid_colon = {
    'n_estimators': [50, 100, 200],
    'max_features': ['sqrt', 'log2'],
    'min_samples_split': [2, 5, 10],
    'max_depth': [None, 10, 20, 30]
}


rf_model_colon = RandomForestClassifier(random_state=42)

rf_grid_search_colon = GridSearchCV(rf_model_colon, rf_param_grid_colon, cv=5, n_jobs=-1, verbose=0, scoring='accuracy')

rf_grid_search_colon.fit(colon_train_normalized, y_train_colon)


# Get the best parameters and score
best_rf_params_colon = rf_grid_search_colon.best_params_
best_rf_score_colon = rf_grid_search_colon.best_score_

print(f"Best Parameters (RF Colon): {best_rf_params_colon}")
print(f"Best Cross-Validation Score (RF Colon): {best_rf_score_colon}")


# Predictions 
best_rf_model_colon = rf_grid_search_colon.best_estimator_
y_pred_rf_colon = best_rf_model_colon.predict(colon_test_normalized)


# Calculate metrics
accuracy_rf_colon = accuracy_score(y_test_colon, y_pred_rf_colon)
f1_rf_colon = f1_score(y_test_colon, y_pred_rf_colon)
recall_rf_colon = recall_score(y_test_colon, y_pred_rf_colon)
precision_rf_colon = precision_score(y_test_colon, y_pred_rf_colon)

print("\nTest Data Metrics (RF Colon)")
print(f"Test Accuracy: {accuracy_rf_colon * 100:.2f}%")
print(f"Test F1 Score: {f1_rf_colon * 100:.2f}%")
print(f"Test Recall: {recall_rf_colon * 100:.2f}%")
print(f"Test Precision: {precision_rf_colon * 100:.2f}%")


# Confusion Matrix Visualization
cm_rf_colon = confusion_matrix(y_test_colon, y_pred_rf_colon)
print("\nConfusion Matrix (RF Colon):")
print(cm_rf_colon)

plt.figure(figsize=(10, 8))
sns.heatmap(cm_rf_colon, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.title('Confusion Matrix - RF Colon')
plt.show()

    
#%% 6.2.1) Classifier with CV 2: SVM (SRBCT)
    

from sklearn.svm import SVC


# Define the parameter grid for SVM
svm_param_grid_srbct = {
    'C': [0.01, 0.1, 1, 5, 10, 50, 100],
    'kernel': ['linear', 'rbf', 'sigmoid'],
    'gamma': ['scale', 'auto', 0.1, 0.5, 1]
}


"""
Low C (0.01-0.3): Strong regularization

High C (5-10): Weak regularization
"""



svm_model_srbct = SVC(random_state=42)

svm_grid_search_srbct = GridSearchCV(svm_model_srbct, svm_param_grid_srbct, cv=5, n_jobs=-1, verbose=0, scoring='accuracy')

svm_grid_search_srbct.fit(srbct_train_normalized, y_train_srbct)


# Get the best parameters and score
best_svm_params_srbct = svm_grid_search_srbct.best_params_
best_svm_score_srbct = svm_grid_search_srbct.best_score_

print(f"Best Parameters (SVM SRBCT): {best_svm_params_srbct}")
print(f"Best Cross-Validation Score (SVM SRBCT): {best_svm_score_srbct}")


# Predictions
best_svm_model_srbct = svm_grid_search_srbct.best_estimator_
y_pred_svm_srbct = best_svm_model_srbct.predict(srbct_test_normalized)


# Calculate metrics
accuracy_svm_srbct = accuracy_score(y_test_srbct, y_pred_svm_srbct)
f1_svm_srbct = f1_score(y_test_srbct, y_pred_svm_srbct, average='micro')
recall_svm_srbct = recall_score(y_test_srbct, y_pred_svm_srbct, average='micro')
precision_svm_srbct = precision_score(y_test_srbct, y_pred_svm_srbct, average='micro')

print("\nTest Data Metrics (SVM SRBCT)")
print(f"Test Accuracy: {accuracy_svm_srbct * 100:.2f}%")
print(f"Test F1 Score: {f1_svm_srbct * 100:.2f}%")
print(f"Test Recall: {recall_svm_srbct * 100:.2f}%")
print(f"Test Precision: {precision_svm_srbct * 100:.2f}%")


# Confusion Matrix Visualization
cm_svm_srbct = confusion_matrix(y_test_srbct, y_pred_svm_srbct)
print("\nConfusion Matrix (SVM SRBCT):")
print(cm_svm_srbct)

plt.figure(figsize=(10, 8))
sns.heatmap(cm_svm_srbct, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.title('Confusion Matrix - SVM SRBCT')
plt.show()


#%% 6.2.2) Classifier with CV 2: SVM (Colon)


svm_param_grid_colon = {
    'C': [0.01, 0.1, 1, 5, 10, 50, 100],
    'kernel': ['linear', 'rbf', 'sigmoid'],
    'gamma': ['scale', 'auto', 0.1, 0.5, 1]
}


svm_model_colon = SVC(random_state=42)

svm_grid_search_colon = GridSearchCV(svm_model_colon, svm_param_grid_colon, cv=5, n_jobs=-1, verbose=0, scoring='accuracy')

svm_grid_search_colon.fit(colon_train_normalized, y_train_colon)


# Get the best parameters and score
best_svm_params_colon = svm_grid_search_colon.best_params_
best_svm_score_colon = svm_grid_search_colon.best_score_


print(f"Best Parameters (SVM Colon): {best_svm_params_colon}")
print(f"Best Cross-Validation Score (SVM Colon): {best_svm_score_colon}")


# Predictions
best_svm_model_colon = svm_grid_search_colon.best_estimator_
y_pred_svm_colon = best_svm_model_colon.predict(colon_test_normalized)


# Calculate metrics
accuracy_svm_colon = accuracy_score(y_test_colon, y_pred_svm_colon)
f1_svm_colon = f1_score(y_test_colon, y_pred_svm_colon)
recall_svm_colon = recall_score(y_test_colon, y_pred_svm_colon)
precision_svm_colon = precision_score(y_test_colon, y_pred_svm_colon)


print("\nTest Data Metrics (SVM Colon)")
print(f"Test Accuracy: {accuracy_svm_colon * 100:.2f}%")
print(f"Test F1 Score: {f1_svm_colon * 100:.2f}%")
print(f"Test Recall: {recall_svm_colon * 100:.2f}%")
print(f"Test Precision: {precision_svm_colon * 100:.2f}%")


# Confusion Matrix Visualization
cm_svm_colon = confusion_matrix(y_test_colon, y_pred_svm_colon)
print("\nConfusion Matrix (SVM Colon):")
print(cm_svm_colon)

plt.figure(figsize=(10, 8))
sns.heatmap(cm_svm_colon, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.title('Confusion Matrix - SVM Colon')
plt.show()



#%% 6.3.1) Classifier with CV 3: LDA (SRBCT)
    


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


# Define the parameter grid for LDA
lda_param_grid_srbct = {
    'solver': ['svd', 'lsqr', 'eigen'],
    'shrinkage': [None, 'auto']             # Only applicable for 'lsqr' solver
}


lda_model_srbct = LinearDiscriminantAnalysis()

lda_grid_search_srbct = GridSearchCV(lda_model_srbct, lda_param_grid_srbct, cv=5, n_jobs=-1, verbose=0, scoring='accuracy')

lda_grid_search_srbct.fit(srbct_train_normalized, y_train_srbct)


# Get the best parameters and score
best_lda_params_srbct = lda_grid_search_srbct.best_params_
best_lda_score_srbct = lda_grid_search_srbct.best_score_


print(f"Best Parameters (LDA SRBCT): {best_lda_params_srbct}")
print(f"Best Cross-Validation Score (LDA SRBCT): {best_lda_score_srbct}")


# Predictions 
best_lda_model_srbct = lda_grid_search_srbct.best_estimator_
y_pred_lda_srbct = best_lda_model_srbct.predict(srbct_test_normalized)


# Calculate metrics
accuracy_lda_srbct = accuracy_score(y_test_srbct, y_pred_lda_srbct)
f1_lda_srbct = f1_score(y_test_srbct, y_pred_lda_srbct, average='micro')
recall_lda_srbct = recall_score(y_test_srbct, y_pred_lda_srbct, average='micro')
precision_lda_srbct = precision_score(y_test_srbct, y_pred_lda_srbct, average='micro')

print("\nTest Data Metrics (LDA SRBCT)")
print(f"Test Accuracy: {accuracy_lda_srbct * 100:.2f}%")
print(f"Test F1 Score: {f1_lda_srbct * 100:.2f}%")
print(f"Test Recall: {recall_lda_srbct * 100:.2f}%")
print(f"Test Precision: {precision_lda_srbct * 100:.2f}%")

# Confusion Matrix Visualization
cm_lda_srbct = confusion_matrix(y_test_srbct, y_pred_lda_srbct)
print("\nConfusion Matrix (LDA SRBCT):")
print(cm_lda_srbct)

plt.figure(figsize=(10, 8))
sns.heatmap(cm_lda_srbct, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.title('Confusion Matrix - LDA SRBCT')
plt.show()



#%% 6.3.2) Classifier with CV 3: LDA (Colon)
    


# Define the parameter grid for LDA
lda_param_grid_colon = {
    'solver': ['svd', 'lsqr', 'eigen'],
    'shrinkage': [None, 'auto']             # Only applicable for 'lsqr' solver
}


lda_model_colon = LinearDiscriminantAnalysis()

lda_grid_search_colon = GridSearchCV(lda_model_colon, lda_param_grid_colon, cv=5, n_jobs=-1, verbose=0, scoring='accuracy')

lda_grid_search_colon.fit(colon_train_normalized, y_train_colon)


# Get the best parameters and score
best_lda_params_colon = lda_grid_search_colon.best_params_
best_lda_score_colon = lda_grid_search_colon.best_score_

print(f"Best Parameters (LDA Colon): {best_lda_params_colon}")
print(f"Best Cross-Validation Score (LDA Colon): {best_lda_score_colon}")


# Predictions 
best_lda_model_colon = lda_grid_search_colon.best_estimator_
y_pred_lda_colon = best_lda_model_colon.predict(colon_test_normalized)


# Calculate metrics
accuracy_lda_colon = accuracy_score(y_test_colon, y_pred_lda_colon)
f1_lda_colon = f1_score(y_test_colon, y_pred_lda_colon)
recall_lda_colon = recall_score(y_test_colon, y_pred_lda_colon)
precision_lda_colon = precision_score(y_test_colon, y_pred_lda_colon)

print("\nTest Data Metrics (LDA Colon)")
print(f"Test Accuracy: {accuracy_lda_colon * 100:.2f}%")
print(f"Test F1 Score: {f1_lda_colon * 100:.2f}%")
print(f"Test Recall: {recall_lda_colon * 100:.2f}%")
print(f"Test Precision: {precision_lda_colon * 100:.2f}%")



# Confusion Matrix Visualization
cm_lda_colon = confusion_matrix(y_test_colon, y_pred_lda_colon)
print("\nConfusion Matrix (LDA Colon):")
print(cm_lda_colon)

plt.figure(figsize=(10, 8))
sns.heatmap(cm_lda_colon, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.title('Confusion Matrix - LDA Colon')
plt.show()


#%% 6.4.1) Classifier with CV 4: XGBoost (SRBCT)
    


from xgboost import XGBClassifier

# Define the parameter grid for XGBoost
xgb_param_grid_srbct = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7, 9],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.5, 0.7, 1.0]
}


xgb_model_srbct = XGBClassifier(eval_metric='mlogloss', random_state=42)

xgb_grid_search_srbct = GridSearchCV(xgb_model_srbct, xgb_param_grid_srbct, cv=5, n_jobs=-1, verbose=0, scoring='accuracy')

xgb_grid_search_srbct.fit(srbct_train_normalized, y_train_srbct)


# Get the best parameters and score
best_xgb_params_srbct = xgb_grid_search_srbct.best_params_
best_xgb_score_srbct = xgb_grid_search_srbct.best_score_

print(f"Best Parameters (XGBoost SRBCT): {best_xgb_params_srbct}")
print(f"Best Cross-Validation Score (XGBoost SRBCT): {best_xgb_score_srbct}")



# Predictions 
best_xgb_model_srbct = xgb_grid_search_srbct.best_estimator_
y_pred_xgb_srbct = best_xgb_model_srbct.predict(srbct_test_normalized)



# Calculate metrics
accuracy_xgb_srbct = accuracy_score(y_test_srbct, y_pred_xgb_srbct)
f1_xgb_srbct = f1_score(y_test_srbct, y_pred_xgb_srbct, average='micro')
recall_xgb_srbct = recall_score(y_test_srbct, y_pred_xgb_srbct, average='micro')
precision_xgb_srbct = precision_score(y_test_srbct, y_pred_xgb_srbct, average='micro')


print("\nTest Data Metrics (XGBoost SRBCT)")
print(f"Test Accuracy: {accuracy_xgb_srbct * 100:.2f}%")
print(f"Test F1 Score: {f1_xgb_srbct * 100:.2f}%")
print(f"Test Recall: {recall_xgb_srbct * 100:.2f}%")
print(f"Test Precision: {precision_xgb_srbct * 100:.2f}%")


# Confusion Matrix Visualization
cm_xgb_srbct = confusion_matrix(y_test_srbct, y_pred_xgb_srbct)
print("\nConfusion Matrix (XGBoost SRBCT):")
print(cm_xgb_srbct)

plt.figure(figsize=(10, 8))
sns.heatmap(cm_xgb_srbct, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.title('Confusion Matrix - XGBoost SRBCT')
plt.show()


#%% 6.4.2) Classifier with CV 4: XGBoost (Colon)
    


xgb_param_grid_colon = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7, 9],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.5, 0.7, 1.0]
}


xgb_model_colon = XGBClassifier(eval_metric='mlogloss', random_state=42)

xgb_grid_search_colon = GridSearchCV(xgb_model_colon, xgb_param_grid_colon, cv=5, n_jobs=-1, verbose=0, scoring='accuracy')

xgb_grid_search_colon.fit(colon_train_normalized, y_train_colon)


# Get the best parameters and score
best_xgb_params_colon = xgb_grid_search_colon.best_params_
best_xgb_score_colon = xgb_grid_search_colon.best_score_

print(f"Best Parameters (XGBoost Colon): {best_xgb_params_colon}")
print(f"Best Cross-Validation Score (XGBoost Colon): {best_xgb_score_colon}")


# Predictions 
best_xgb_model_colon = xgb_grid_search_colon.best_estimator_
y_pred_xgb_colon = best_xgb_model_colon.predict(colon_test_normalized)



# Calculate metrics
accuracy_xgb_colon = accuracy_score(y_test_colon, y_pred_xgb_colon)
f1_xgb_colon = f1_score(y_test_colon, y_pred_xgb_colon)
recall_xgb_colon = recall_score(y_test_colon, y_pred_xgb_colon)
precision_xgb_colon = precision_score(y_test_colon, y_pred_xgb_colon)


print("\nTest Data Metrics (XGBoost Colon)")
print(f"Test Accuracy: {accuracy_xgb_colon * 100:.2f}%")
print(f"Test F1 Score: {f1_xgb_colon * 100:.2f}%")
print(f"Test Recall: {recall_xgb_colon * 100:.2f}%")
print(f"Test Precision: {precision_xgb_colon * 100:.2f}%")



# Confusion Matrix Visualization
cm_xgb_colon = confusion_matrix(y_test_colon, y_pred_xgb_colon)
print("\nConfusion Matrix (XGBoost Colon):")
print(cm_xgb_colon)

plt.figure(figsize=(10, 8))
sns.heatmap(cm_xgb_colon, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.title('Confusion Matrix - XGBoost Colon')
plt.show()



#%% 6.5.1) Classifier with CV 5: MLP (SRBCT)



from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (accuracy_score, f1_score, recall_score, 
                             precision_score, confusion_matrix, classification_report)



# 1) One-hot Encoding
num_classes = 4  
y_train_srbct_onehot = to_categorical(y_train_srbct, num_classes=num_classes)
y_test_srbct_onehot = to_categorical(y_test_srbct, num_classes=num_classes)


# 2) Model definition
act_function = 'tanh'
batch = 16

def create_mlp_model(input_shape):
    model = Sequential([
        Dense(50, activation=act_function, input_shape=(input_shape,)),
        BatchNormalization(),
        Dropout(0.3),
        Dense(100, activation=act_function, kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Dropout(0.3),   
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model



# 3) Cross-Validation
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = []
fold_number = 1


print("\nStarting 5-Fold Cross-Validation...")

for train_idx, val_idx in kfold.split(srbct_train_normalized, y_train_srbct):
    
    print(f"Training Fold {fold_number}/5")  
    
    
    # Split data
    x_train_fold = srbct_train_normalized[train_idx]  
    y_train_fold = y_train_srbct_onehot[train_idx]    
    x_val_fold = srbct_train_normalized[val_idx]      
    y_val_fold = y_train_srbct_onehot[val_idx]        
    
    # Create fresh model
    model = create_mlp_model(input_shape=x_train_fold.shape[1])
    
    # Define callbacks for this fold
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=20,
        restore_best_weights=True,
        verbose=0
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=10,
        min_lr=1e-7,
        verbose=0
    )
    
    # Train
    history = model.fit(
        x_train_fold, y_train_fold,
        validation_data=(x_val_fold, y_val_fold),
        epochs=50,
        batch_size=batch,  
        callbacks=[early_stopping, reduce_lr],
        verbose=0
    )
    
    # Evaluate
    scores = model.evaluate(x_val_fold, y_val_fold, verbose=0)
    cv_scores.append(scores[1] * 100)
    
    print(f"Fold {fold_number} Validation Accuracy: {scores[1] * 100:.2f}%")
    fold_number += 1


# Print CV results
print("\nCross-Validation Results:")
print(f"Mean CV Accuracy: {np.mean(cv_scores):.2f}%")
print(f"Std CV Accuracy: {np.std(cv_scores):.2f}%")
print(f"All Fold Scores: {[f'{score:.2f}%' for score in cv_scores]}")




# 4) Train final model

print("\nTraining Final Model on Full Training Data...")


final_model = create_mlp_model(input_shape=srbct_train_normalized.shape[1])


# Define fresh callbacks
final_early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=20,
    restore_best_weights=True,
    verbose=1
)

final_reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=10,
    min_lr=1e-7,
    verbose=1
)


# Training
final_history = final_model.fit(
    srbct_train_normalized, y_train_srbct_onehot,  
    validation_split=0.2,
    epochs=50,
    batch_size=batch,  
    callbacks=[final_early_stopping, final_reduce_lr],
    verbose=1  
)


# Predictions
y_predicted_proba = final_model.predict(srbct_test_normalized)
y_predicted_mlp_encoded = np.argmax(y_predicted_proba, axis=1)


# 5) Evaluation

print("\nTest Data Metrics")

accuracy = accuracy_score(y_test_srbct, y_predicted_mlp_encoded) 
f1 = f1_score(y_test_srbct, y_predicted_mlp_encoded, average='micro')  
recall = recall_score(y_test_srbct, y_predicted_mlp_encoded, average='micro')  
precision = precision_score(y_test_srbct, y_predicted_mlp_encoded, average='micro')  

print(f"Test Accuracy: {accuracy * 100:.2f}%")  
print(f"Test F1 Score (micro): {f1 * 100:.2f}%")
print(f"Test Recall (micro): {recall * 100:.2f}%")
print(f"Test Precision (micro): {precision * 100:.2f}%")


print("\nDetailed Classification Report:")
print(classification_report(y_test_srbct, y_predicted_mlp_encoded))



# 6) Confusion Matrix

cm_test = confusion_matrix(y_test_srbct, y_predicted_mlp_encoded)  
print("\nConfusion Matrix:")
print(cm_test)

plt.figure(figsize=(10, 8))
sns.heatmap(cm_test, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix - MLP')
plt.show()



# 7) Training history
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

axes[0].plot(final_history.history['accuracy'], label='Training Accuracy')
axes[0].plot(final_history.history['val_accuracy'], label='Validation Accuracy')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy')
axes[0].set_title('Model Accuracy over Epochs')
axes[0].legend()
axes[0].grid(True)

axes[1].plot(final_history.history['loss'], label='Training Loss')
axes[1].plot(final_history.history['val_loss'], label='Validation Loss')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].set_title('Model Loss over Epochs')
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.show()




#%% 6.5.2) Classifier with CV 5: MLP (Colon)



from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (accuracy_score, f1_score, recall_score, 
                             precision_score, confusion_matrix, classification_report)



# 1) One-hot Encoding
num_classes_colon = 2  
y_train_colon_onehot = to_categorical(y_train_colon, num_classes=num_classes_colon)
y_test_colon_onehot = to_categorical(y_test_colon, num_classes=num_classes_colon)


# 2) Model definition
act_function = 'tanh'
batch = 8

def create_mlp_model_colon(input_shape):
    model = Sequential([
        Dense(50, activation=act_function, input_shape=(input_shape,)),
        BatchNormalization(),
        Dropout(0.3),
        Dense(100, activation=act_function, kernel_regularizer=l2(0.01)),
        BatchNormalization(),
        Dropout(0.3),   
        Dense(num_classes_colon, activation='softmax')  # 2 neurons for binary classification
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model




# 3) Cross-Validation

kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores_colon = []
fold_number = 1


print("\nStarting 5-Fold Cross-Validation...")

for train_idx, val_idx in kfold.split(colon_train_normalized, y_train_colon):
    
    print(f"Training Fold {fold_number}/5")  
    
    
    # Split data
    x_train_fold = colon_train_normalized[train_idx]  
    y_train_fold = y_train_colon_onehot[train_idx]    
    x_val_fold = colon_train_normalized[val_idx]      
    y_val_fold = y_train_colon_onehot[val_idx]        
    
    # Create fresh model
    model = create_mlp_model_colon(input_shape=x_train_fold.shape[1])
    
    # Define callbacks for this fold
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=20,
        restore_best_weights=True,
        verbose=0
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=10,
        min_lr=1e-7,
        verbose=0
    )
    
    # Train
    history = model.fit(
        x_train_fold, y_train_fold,
        validation_data=(x_val_fold, y_val_fold),
        epochs=50,
        batch_size=batch,  
        callbacks=[early_stopping, reduce_lr],
        verbose=0
    )
    
    # Evaluate
    scores = model.evaluate(x_val_fold, y_val_fold, verbose=0)
    cv_scores_colon.append(scores[1] * 100)
    
    print(f"Fold {fold_number} Validation Accuracy: {scores[1] * 100:.2f}%")
    fold_number += 1



# Print CV results
print("\nCross-Validation Results:")
print(f"Mean CV Accuracy: {np.mean(cv_scores_colon):.2f}%")
print(f"Std CV Accuracy: {np.std(cv_scores_colon):.2f}%")
print(f"All Fold Scores: {[f'{score:.2f}%' for score in cv_scores_colon]}")





# 4) Train final model

print("\nTraining Final Model on Full Training Data...")


final_model_colon = create_mlp_model_colon(input_shape=colon_train_normalized.shape[1])


# Define fresh callbacks
final_early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=20,
    restore_best_weights=True,
    verbose=1
)

final_reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=10,
    min_lr=1e-7,
    verbose=1
)


# Training
final_history_colon = final_model_colon.fit(
    colon_train_normalized, y_train_colon_onehot,  
    validation_split=0.2,
    epochs=50,
    batch_size=batch,  
    callbacks=[final_early_stopping, final_reduce_lr],
    verbose=1  
)


# Predictions
y_predicted_proba_colon = final_model_colon.predict(colon_test_normalized)
y_predicted_mlp_colon = np.argmax(y_predicted_proba_colon, axis=1)




# 5) Evaluation

print("\nTest Data Metrics (MLP Colon)")

accuracy_mlp_colon = accuracy_score(y_test_colon, y_predicted_mlp_colon) 
f1_mlp_colon = f1_score(y_test_colon, y_predicted_mlp_colon)  
recall_mlp_colon = recall_score(y_test_colon, y_predicted_mlp_colon)  
precision_mlp_colon = precision_score(y_test_colon, y_predicted_mlp_colon)  

print(f"Test Accuracy: {accuracy_mlp_colon * 100:.2f}%")  
print(f"Test F1 Score: {f1_mlp_colon * 100:.2f}%")
print(f"Test Recall: {recall_mlp_colon * 100:.2f}%")
print(f"Test Precision: {precision_mlp_colon * 100:.2f}%")


print("\nDetailed Classification Report:")
print(classification_report(y_test_colon, y_predicted_mlp_colon))




# 6) Confusion Matrix

cm_test_colon = confusion_matrix(y_test_colon, y_predicted_mlp_colon)  
print("\nConfusion Matrix (MLP Colon):")
print(cm_test_colon)

plt.figure(figsize=(10, 8))
sns.heatmap(cm_test_colon, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix - MLP Colon')
plt.show()




# 7) Training history
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

axes[0].plot(final_history_colon.history['accuracy'], label='Training Accuracy')
axes[0].plot(final_history_colon.history['val_accuracy'], label='Validation Accuracy')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy')
axes[0].set_title('Model Accuracy over Epochs (MLP Colon)')
axes[0].legend()
axes[0].grid(True)

axes[1].plot(final_history_colon.history['loss'], label='Training Loss')
axes[1].plot(final_history_colon.history['val_loss'], label='Validation Loss')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].set_title('Model Loss over Epochs (MLP Colon)')
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.show()




#%% 7) Analysis of the results

#1 model results bar graph

#2 correlation matirx of selected features for 3 methods



#%% 8) Resultant metrics and models


"""
Mutual Information
-------------------------
Best Parameters (RF SRBCT): {'max_depth': None, 'max_features': 'sqrt', 'min_samples_split': 2, 'n_estimators': 50}
Best Cross-Validation Score (RF SRBCT): 1.0

Test Data Metrics (RF SRBCT)
Test Accuracy: 100.00%
Test F1 Score: 100.00%
Test Recall: 100.00%
Test Precision: 100.00%

Best Parameters (RF Colon): {'max_depth': None, 'max_features': 'sqrt', 'min_samples_split': 2, 'n_estimators': 50}
Best Cross-Validation Score (RF Colon): 0.9199999999999999

Test Data Metrics (RF Colon)
Test Accuracy: 76.92%
Test F1 Score: 80.00%
Test Recall: 75.00%
Test Precision: 85.71%


Best Parameters (SVM SRBCT): {'C': 0.01, 'gamma': 'scale', 'kernel': 'linear'}
Best Cross-Validation Score (SVM SRBCT): 1.0

Test Data Metrics (SVM SRBCT)
Test Accuracy: 100.00%
Test F1 Score: 100.00%
Test Recall: 100.00%
Test Precision: 100.00%

Best Parameters (SVM Colon): {'C': 5, 'gamma': 'scale', 'kernel': 'rbf'}
Best Cross-Validation Score (SVM Colon): 0.9399999999999998

Test Data Metrics (SVM Colon)
Test Accuracy: 76.92%
Test F1 Score: 80.00%
Test Recall: 75.00%
Test Precision: 85.71%


Best Parameters (LDA SRBCT): {'shrinkage': 'auto', 'solver': 'lsqr'}
Best Cross-Validation Score (LDA SRBCT): 1.0

Test Data Metrics (LDA SRBCT)
Test Accuracy: 100.00%
Test F1 Score: 100.00%
Test Recall: 100.00%
Test Precision: 100.00%

Best Parameters (LDA Colon): {'shrinkage': 'auto', 'solver': 'lsqr'}
Best Cross-Validation Score (LDA Colon): 0.9199999999999999

Test Data Metrics (LDA Colon)
Test Accuracy: 84.62%
Test F1 Score: 87.50%
Test Recall: 87.50%
Test Precision: 87.50%


Best Parameters (XGBoost SRBCT): {'learning_rate': 0.2, 'max_depth': 3, 'n_estimators': 50, 'subsample': 0.5}
Best Cross-Validation Score (XGBoost SRBCT): 0.956043956043956

Test Data Metrics (XGBoost SRBCT)
Test Accuracy: 100.00%
Test F1 Score: 100.00%
Test Recall: 100.00%
Test Precision: 100.00%

Best Parameters (XGBoost Colon): {'learning_rate': 0.2, 'max_depth': 3, 'n_estimators': 100, 'subsample': 0.5}
Best Cross-Validation Score (XGBoost Colon): 0.8955555555555555

Test Data Metrics (XGBoost Colon)
Test Accuracy: 76.92%
Test F1 Score: 80.00%
Test Recall: 75.00%
Test Precision: 85.71%


MLP model for SRBCT
5 fold Cross-Validation Results:
Mean CV Accuracy: 100.00%
Std CV Accuracy: 0.00%

Test Data Metrics
Test Accuracy: 100.00%
Test F1 Score (micro): 100.00%
Test Recall (micro): 100.00%
Test Precision (micro): 100.00%

Model Architecture
act_function = 'tanh'
Neurons: 50 (input layer) - 100 (hidden layer) - 4 (softmax layer)
Droput: 0.3
L2 regularizer: 0.001
Optimizer= Adam
Learning rate = 0.001
Learning rate scheduling: "ReduceLROnPlateau" with patience = 10
Loss Function = categorical crossentropy
batch size: 16
epochs: 50 (with early stopping patience=20)

MLP model for Colon
5 fold Cross-Validation Results:
Mean CV Accuracy: 89.78%
Std CV Accuracy: 6.34%

Test Data Metrics (MLP Colon)
Test Accuracy: 84.62%
Test F1 Score: 87.50%
Test Recall: 87.50%
Test Precision: 87.50%

Model Architecture
act_function = 'tanh'
Neurons: 50 (input layer) - 100 (hidden layer) - 2 (softmax layer)
Droput: 0.3
L2 regularizer: 0.01
Optimizer= Adam
Learning rate = 0.001
Learning rate scheduling: "ReduceLROnPlateau" with patience = 10
Loss Function = binary crossentropy
batch size: 8
epochs: 50 (with early stopping patience=20)
"""



"""
RFE
-------------------------

Best Parameters (RF SRBCT): {'max_depth': None, 'max_features': 'sqrt', 'min_samples_split': 2, 'n_estimators': 50}
Best Cross-Validation Score (RF SRBCT): 1.0

Test Data Metrics (RF SRBCT)
Test Accuracy: 100.00%
Test F1 Score: 100.00%
Test Recall: 100.00%
Test Precision: 100.00%


Best Parameters (RF Colon): {'max_depth': None, 'max_features': 'sqrt', 'min_samples_split': 10, 'n_estimators': 50}
Best Cross-Validation Score (RF Colon): 0.86

Test Data Metrics (RF Colon)
Test Accuracy: 76.92%
Test F1 Score: 80.00%
Test Recall: 75.00%
Test Precision: 85.71%


Best Parameters (SVM SRBCT): {'C': 0.01, 'gamma': 'scale', 'kernel': 'linear'}
Best Cross-Validation Score (SVM SRBCT): 1.0

Test Data Metrics (SVM SRBCT)
Test Accuracy: 100.00%
Test F1 Score: 100.00%
Test Recall: 100.00%
Test Precision: 100.00%


Best Parameters (SVM Colon): {'C': 0.1, 'gamma': 'scale', 'kernel': 'linear'}
Best Cross-Validation Score (SVM Colon): 0.9199999999999999

Test Data Metrics (SVM Colon)
Test Accuracy: 76.92%
Test F1 Score: 82.35%
Test Recall: 87.50%
Test Precision: 77.78%


Best Parameters (LDA SRBCT): {'shrinkage': 'auto', 'solver': 'lsqr'}
Best Cross-Validation Score (LDA SRBCT): 1.0

Test Data Metrics (LDA SRBCT)
Test Accuracy: 100.00%
Test F1 Score: 100.00%
Test Recall: 100.00%
Test Precision: 100.00%


Best Parameters (LDA Colon): {'shrinkage': 'auto', 'solver': 'lsqr'}
Best Cross-Validation Score (LDA Colon): 0.9199999999999999

Test Data Metrics (LDA Colon)
Test Accuracy: 84.62%
Test F1 Score: 87.50%
Test Recall: 87.50%
Test Precision: 87.50%


Best Parameters (XGBoost SRBCT): {'learning_rate': 0.01, 'max_depth': 3, 'n_estimators': 50, 'subsample': 0.5}
Best Cross-Validation Score (XGBoost SRBCT): 0.956043956043956

Test Data Metrics (XGBoost SRBCT)
Test Accuracy: 100.00%
Test F1 Score: 100.00%
Test Recall: 100.00%
Test Precision: 100.00%

Best Parameters (XGBoost Colon): {'learning_rate': 0.2, 'max_depth': 3, 'n_estimators': 50, 'subsample': 0.5}
Best Cross-Validation Score (XGBoost Colon): 0.8377777777777778

Test Data Metrics (XGBoost Colon)
Test Accuracy: 84.62%
Test F1 Score: 87.50%
Test Recall: 87.50%
Test Precision: 87.50%


MLP SRBCT

Cross-Validation Results:
Mean CV Accuracy: 98.57%
Std CV Accuracy: 2.86%

Test Data Metrics
Test Accuracy: 100.00%
Test F1 Score (micro): 100.00%
Test Recall (micro): 100.00%
Test Precision (micro): 100.00%

model parameters are same as above SRBCT MLP model.


MLP COLON

Cross-Validation Results:
Mean CV Accuracy: 94.00%
Std CV Accuracy: 4.90%

Test Data Metrics (MLP Colon)
Test Accuracy: 76.92%
Test F1 Score: 82.35%
Test Recall: 87.50%
Test Precision: 77.78%

model parameters are same as above Colon MLP model.

"""



"""
LASSO
-------------------------







MLP Colon

Cross-Validation Results:
Mean CV Accuracy: 95.78%
Std CV Accuracy: 5.18%

Test Data Metrics (MLP Colon)
Test Accuracy: 84.62%
Test F1 Score: 87.50%
Test Recall: 87.50%
Test Precision: 87.50%

"""

