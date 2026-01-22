# Classification of Gene Expression Levels by Microarray Datasets

This repository implements an end-to-end classification pipeline for benchmark microarray gene-expression datasets. It focuses on selecting a compact set of informative genes and comparing multiple classifiers under cross-validation and held-out testing. <br><br>

## Datasets

Two datasets are included in ARFF format:

- **SRBCT**: Multi-class classification (4 tumor types). File: `SRBCT.arff`. 
83 instances with 2308 features (genes) and class information. <br>
        
- **Colon Tumor**: Binary classification (Tumor vs Normal). File: `Colon.arff`.
62 instances with 2000 features, and binary class information. <br>


Dataset source: https://csse.szu.edu.cn/staff/zhuzx/Datasets.html

Reference paper: Zhu et al., 2007, *“Markov blanket-embedded genetic algorithm for gene selection”*. <br><br>

## Methodology 

<br>

### 1) Data loading and label handling

- **SRBCT** is loaded using `scipy.io.arff.loadarff` and class labels are decoded and converted to integers.
- **Colon Tumor** is parsed manually because the ARFF header contains duplicate gene identifiers; duplicates are disambiguated by appending suffixes (e.g., `R39465_1`, `R39465_2`). Labels are mapped to **Tumor = 1**, **Normal = 0**. <br><br>

### 2) Exploratory data analysis (EDA)

Standard microarray EDA visualizations are generated for each dataset, including class distribution, per-gene box/violin plots, histograms, summary-statistic heatmaps, and correlation heatmaps. <br><br>

### 3) Train/test split 

Data are split into **80% training** and **20% test** sets before feature selection and normalization to reduce data leakage risk. <br><br>

### 4) Feature selection (you need to run one of them)

Feature selection is run on the training set and then applied to the test set:

- **Mutual Information**: selects the top 50 genes.
- **RFE (Recursive Feature Elimination)** with Logistic Regression: selects 50 genes.
- **LASSO (LassoCV)**: ranks coefficients and keeps the top 50 genes by absolute magnitude. <br><br>

### 5) Normalization

Selected gene-expression features are standardized with `StandardScaler`, fit **only on the training set**, and then applied to both training and test data. <br><br>

### 6) Modeling and evaluation

The pipeline compares five model families:

- Random Forest (RF)
- Support Vector Machine (SVM)
- Linear Discriminant Analysis (LDA)
- XGBoost
- Multi-Layer Perceptron (MLP; Keras/TensorFlow)

For RF/SVM/LDA/XGBoost, **5-fold cross-validation** is performed via `GridSearchCV` for hyperparameter selection. For the MLP, **5-fold StratifiedKFold** is performed with early stopping and learning-rate scheduling, then a final model is trained on the full training split.

Evaluation includes **accuracy**, **F1**, **precision**, **recall**, and **confusion matrices**, reported on the held-out test set.

<br><br>

## Results

The following results are reported as **accuracy (%)** under **5-fold cross-validation (CV)** and on the **held-out test set** after selecting 50 genes. Values are taken from the performed experiments in this repository (the exact numbers can vary with different splits, seeds, and library versions). <br><br>

### SRBCT (4-class)

| Feature selection | RF (CV/Test) | SVM (CV/Test) | LDA (CV/Test) | XGBoost (CV/Test) | MLP (CV/Test) |
|---|---:|---:|---:|---:|---:|
| Mutual Information | 100.0 / 100.0 | 100.0 / 100.0 | 100.0 / 100.0 | 95.6 / 100.0 | 100.0 / 100.0 |
| RFE | 100.0 / 100.0 | 100.0 / 100.0 | 100.0 / 100.0 | 95.6 / 100.0 | 98.6 / 100.0 |
| LASSO | 96.9 / 100.0 | 98.5 / 94.1 | 96.9 / 88.2 | 93.9 / 94.1 | 95.4 / 94.1 |

Key observation: for SRBCT, perfect test accuracy is achieved for all models under Mutual Information and RFE in the reported runs; LASSO yields lower test accuracy for several models. <br><br>

### Colon Tumor (binary)

| Feature selection | RF (CV/Test) | SVM (CV/Test) | LDA (CV/Test) | XGBoost (CV/Test) | MLP (CV/Test) |
|---|---:|---:|---:|---:|---:|
| Mutual Information | 92.0 / 76.9 | 94.0 / 76.9 | 92.0 / 84.6 | 89.6 / 76.9 | 89.8 / 84.6 |
| RFE | 86.0 / 76.9 | 92.0 / 76.9 | 92.0 / 84.6 | 83.8 / 84.6 | 94.0 / 76.9 |
| LASSO | 83.6 / 76.9 | 91.8 / 69.2 | 87.6 / 84.6 | 87.8 / 69.2 | 91.8 / 76.9 |

Key observation: for Colon Tumor, the best reported test accuracy is **84.6%**, most consistently reached by **LDA** across all three feature-selection approaches. <br><br>

## Outputs

- Figures and confusion matrices are generated during execution.
- Example figures and result screenshots are provided under `analyses/` (EDA plots, feature-selection plots, and summary comparisons). <br><br>

## How to run

1. Ensure `SRBCT.arff` and `Colon.arff` are in the working directory.
2. Run `gene_expression_levels.py` **cell-by-cell** (it is organized with `#%%` sections for an IDE workflow similar to a notebook). <br><br>

### Dependencies

Core: `numpy`, `pandas`, `scipy`, `matplotlib`, `seaborn`

ML: `scikit-learn`, `xgboost`, `tensorflow` (Keras) <br><br>

## Notes

- The Colon dataset contains duplicate gene names in the ARFF header; duplicates are resolved by creating unique column names.
- Reported results can vary slightly with different train/test splits or random seeds.


<br><br>
:mailbox: Contact me at memisoguzhants@gmail.com <br> I'm waiting for your suggestions and contributions.