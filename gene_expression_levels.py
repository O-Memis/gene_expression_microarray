"""
Classification of Gene Expression Levels from a benchmark Microarray dataset

Oguzhan Memis             12/01/2026


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


# Plotting class distribution
plt.figure(figsize=(15, 10))
sns.countplot(x=class_col, data=df_srbct, palette='Set2', hue=class_col, legend=False)
plt.title('SRBCT Class Distribution')
plt.xlabel('Cancer Type')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()


# Plotting boxplots for gene expression of the first 20 selected genes
selected_genes = df_srbct.columns[:-1][:20]  
plt.figure(figsize=(20, 15))  # Adjusted size for more subplots
for i, gene in enumerate(selected_genes):
    plt.subplot(5, 4, i + 1)  # 5 rows and 4 columns for 20 genes
    sns.boxplot(x=class_col, y=gene, data=df_srbct, palette='Set2', hue=class_col, legend=False)
    plt.title(f'Expression of {gene} by Class')
plt.tight_layout()
plt.show()


# Plotting histograms for gene expression distribution
plt.figure(figsize=(20, 15))  # Adjusted size for more subplots
for i, gene in enumerate(selected_genes):
    plt.subplot(5, 4, i + 1)  # 5 rows and 4 columns for 20 genes
    sns.histplot(df_srbct[gene], bins=30, kde=True, color='blue')
    plt.title(f'Histogram of {gene} Expression')
plt.tight_layout()
plt.show()


# Correlation heatmap for the first few genes
correlation_matrix = df_srbct[selected_genes].corr()
plt.figure(figsize=(15, 10))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='Purples', square=True, vmin=-0.5, vmax=1)
plt.title('Correlation Heatmap for Selected Genes (SRBCT)')
plt.show()


# Plotting violin plots for gene expression of the first 20 selected genes
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

# Plotting class distribution
plt.figure(figsize=(8, 5))
sns.countplot(x='class', data=df_colon, palette='Set2', hue='class', legend=False)
plt.title('Colon Class Distribution')
plt.xlabel('Tumor (1) vs Normal (0)')
plt.ylabel('Count')
plt.xticks(ticks=[0, 1], labels=['Normal', 'Tumor'])
plt.show()


# Plotting boxplots for gene expression of the first 20 selected genes
selected_genes_colon = df_colon.columns[:-1][:20]  
plt.figure(figsize=(20, 15))  # Adjusted size for more subplots
for i, gene in enumerate(selected_genes_colon):
    plt.subplot(5, 4, i + 1)  # 5 rows and 4 columns for 20 genes
    sns.boxplot(x='class', y=gene, data=df_colon, palette='Set2', hue='class', legend=False)
    plt.title(f'Expression of {gene} by Class')
plt.tight_layout()
plt.show()


# Plotting histograms for gene expression distribution
plt.figure(figsize=(20, 15))  # Adjusted size for more subplots
for i, gene in enumerate(selected_genes_colon):
    plt.subplot(5, 4, i + 1)  # 5 rows and 4 columns for 20 genes
    sns.histplot(df_colon[gene], bins=30, kde=True, color='green')
    plt.title(f'Histogram of {gene} Expression')
plt.tight_layout()
plt.show()


# Correlation heatmap for the first few genes
correlation_matrix_colon = df_colon[selected_genes_colon].corr()
plt.figure(figsize=(15, 10))
sns.heatmap(correlation_matrix_colon, annot=True, fmt=".2f", cmap='Purples', square=True, vmin=-0.5, vmax=1)
plt.title('Correlation Heatmap for Selected Genes (Colon)')
plt.show()


# Plotting violin plots for gene expression of the first 20 selected genes
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
plt.figure(figsize=(10, 6))
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
plt.figure(figsize=(10, 6))
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



# 1) RFE for SRBCT Training Set
model_srbct = LogisticRegression(max_iter=1000)
rfe_srbct = RFE(estimator=model_srbct, n_features_to_select=50)
rfe_srbct.fit(X_train_srbct, y_train_srbct)


# Get the selected features for SRBCT
selected_features_rfe_srbct = X_train_srbct.columns[rfe_srbct.support_]
print("\n--- Selected Features by RFE (SRBCT) ---\n")
print(selected_features_rfe_srbct)


# Plotting the ranking of features for SRBCT
plt.figure(figsize=(10, 6))
sns.barplot(x=X_train_srbct.columns, y=rfe_srbct.ranking_, palette='Set2')
plt.title('Feature Ranking by RFE (SRBCT)')
plt.xlabel('Features')
plt.ylabel('Ranking')
plt.xticks(rotation=45)
plt.show()




# 2) RFE for Colon Training Set
model_colon = LogisticRegression(max_iter=1000)
rfe_colon = RFE(estimator=model_colon, n_features_to_select=50)
rfe_colon.fit(X_train_colon, y_train_colon)


# Get the selected features for Colon
selected_features_rfe_colon = X_train_colon.columns[rfe_colon.support_]
print("\n--- Selected Features by RFE (Colon) ---\n")
print(selected_features_rfe_colon)


# Plotting the ranking of features for Colon
plt.figure(figsize=(10, 6))
sns.barplot(x=X_train_colon.columns, y=rfe_colon.ranking_, palette='Set2')
plt.title('Feature Ranking by RFE (Colon)')
plt.xlabel('Features')
plt.ylabel('Ranking')
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
selected_features_lasso_srbct = X_train_srbct.columns[lasso_coef_srbct != 0]


print("\n--- Selected Features by Lasso Regression (SRBCT) ---\n")
print(selected_features_lasso_srbct)


# Plotting the coefficients of the Lasso model for SRBCT
plt.figure(figsize=(10, 6))
sns.barplot(x=X_train_srbct.columns, y=lasso_coef_srbct, palette='Set2')
plt.title('Lasso Coefficients (SRBCT)')
plt.xlabel('Features')
plt.ylabel('Coefficient Value')
plt.xticks(rotation=45)
plt.axhline(0, color='black', linestyle='--')
plt.show()




# 2) Lasso Regression for Colon Training Set
lasso_colon = LassoCV(alphas=np.logspace(-4, 4, 100), cv=5, random_state=42)
lasso_colon.fit(X_train_colon, y_train_colon)


# Get the coefficients and selected features for Colon
lasso_coef_colon = lasso_colon.coef_
selected_features_lasso_colon = X_train_colon.columns[lasso_coef_colon != 0]


print("\n--- Selected Features by Lasso Regression (Colon) ---\n")
print(selected_features_lasso_colon)


# Plotting the coefficients of the Lasso model for Colon
plt.figure(figsize=(10, 6))
sns.barplot(x=X_train_colon.columns, y=lasso_coef_colon, palette='Set2')
plt.title('Lasso Coefficients (Colon)')
plt.xlabel('Features')
plt.ylabel('Coefficient Value')
plt.xticks(rotation=45)
plt.axhline(0, color='black', linestyle='--')
plt.show()



# Transforming the training sets to include only the selected features from Lasso
X_train_srbct_selected = X_train_srbct[selected_features_lasso_srbct]
X_train_colon_selected = X_train_colon[selected_features_lasso_colon]

# Then apply this selection to the test sets
X_test_srbct_selected = X_test_srbct[selected_features_lasso_srbct]
X_test_colon_selected = X_test_colon[selected_features_lasso_colon]

print(f"Transformed SRBCT Training set shape (Lasso): {X_train_srbct_selected.shape}")
print(f"Transformed SRBCT Test set shape (Lasso): {X_test_srbct_selected.shape}")
print(f"Transformed Colon Training set shape (Lasso): {X_train_colon_selected.shape}")
print(f"Transformed Colon Test set shape (Lasso): {X_test_colon_selected.shape}")




#%% 5) Normalization


"""
Run one of the 3 feature selection methods to compare them.
"""


# From dataframe to Numpy arrays for simplicity
x_train_srbct_selected = X_train_srbct_selected.to_numpy()
x_train_colon_selected = X_train_colon_selected.to_numpy()


x_test_srbct_selected = X_test_srbct_selected.to_numpy()
x_test_colon_selected = X_test_colon_selected.to_numpy()








#%% 6.1.1) Classifier with CV 1: RF (SRBCT)
    
#%% 6.1.2) Classifier with CV 1: RF (Colon)
    
#%% 6.2.1) Classifier with CV 2: SVM (SRBCT)
    
#%% 6.2.2) Classifier with CV 2: SVM (Colon)

#%% 6.3.1) Classifier with CV 3: LDA (SRBCT)
    
#%% 6.3.2) Classifier with CV 3: LDA (Colon)
    
#%% 6.4.1) Classifier with CV 4: XGBoost (SRBCT)
    
#%% 6.4.2) Classifier with CV 4: XGBoost (Colon)
    
#%% 6.5.1) Classifier with CV 5: MLP (SRBCT)
    
#%% 6.5.2) Classifier with CV 5: MLP (Colon)
    

#%% 7) Analysis of the results



#%% 8) Resultant metrics and models

