#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import time
import numpy as np
import pandas as pd
import seaborn as sns
import ast
import re
import matplotlib.pyplot as plt
import xarray as xr

from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, StackingClassifier, GradientBoostingClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, auc, classification_report, cohen_kappa_score, 
                             f1_score, log_loss, precision_recall_fscore_support, 
                             precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import (cross_val_score, GridSearchCV, KFold, RepeatedStratifiedKFold, 
                                     StratifiedShuffleSplit, train_test_split)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_sample_weight

from imblearn.over_sampling import SMOTE, SVMSMOTE

from xgboost import XGBClassifier
import xgboost as xgb

import shap
from tabulate import tabulate
from functools import *
from ydata_profiling import ProfileReport
from IPython.display import HTML
from numpy import interp


rgpath = "path/to/your/forest/"
X = pd.read_csv('path/to/your/original_processed_data.csv')
y1 = np.loadtxt('path/to/your/depression_status_test.csv', delimiter=',', dtype=int)
results_file_path = os.path.join(rgpath, "forest_ablation_study_results_v1.csv")
best_config_file_path = os.path.join(rgpath, "forest_ablation_best_config_v1.csv")
fig_outfile_path = = os.path.join(rgpath, "forest_ablation_test_hyperpara_heatmap.pdf")


if isinstance(X, pd.DataFrame):
    X = X.values

def update_results(file_path, config, metrics):
    new_row = pd.DataFrame([{**{"Configuration": config}, **metrics}])
    with open(file_path, 'a') as f:
        new_row.to_csv(f, header=f.tell()==0, index=False)

def composite_score(row, weight_auc=2, weight_kappa=0.25):
    recall = row['Recall']
    cs = weight_auc * row['roc_auc'] + weight_kappa * row['Cohen\'s Kappa']
    return cs * (recall > 0)    


# Define your base configurations and ablation parameters

base_configs = {
    "Random Forest": {"class_weight": "balanced", "random_state": 42,"n_jobs": 18,"criterion":"gini"},
    "XGBoost": {"objective": 'binary:logistic', "colsample_bytree": 0.3, "max_depth": 4, "alpha": 10, "n_estimators": 1000,"n_jobs": 18,"learning_rate": 0.1},
    "Logistic Regression": {"max_iter": 5000, "solver": 'lbfgs', "class_weight": 'balanced'},
    "Naive Bayes": {}
}



rf_params = [
    {"n_estimators": ne, "max_depth": md, "min_samples_split": sp, "min_samples_leaf": lf}
    for ne in [1000]
    for md in [10,11,12,13]
    for sp in [8,10,12]
    for lf in [8,10,12]
]


ablation_params = { "Random Forest": rf_params}


n_splits = 10  # Number of folds
kf = KFold(n_splits=n_splits, shuffle=True)

# Define the function for ablation study with cross-validation
def ablation_study_cross_val(X, y, model, model_name, config, kf):
    scores = {
        "Accuracy": [],
        "Recall": [],
        "F1 Weighted": [],
        "Cohen's Kappa": [],
        "Positive Precision": [],
        "Negative Precision": [],
        "Error Rate": [],
        "Loss": [],
        "roc_auc": [],
        "Composite Score": []
    }
    
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        # Apply SMOTE for balancing the dataset
        #smote = SMOTE(k_neighbors=11)
        #X_train, y_train = smote.fit_resample(X_train, y_train)
        
        # Fit the model
        model.set_params(**config)
        # Fit the model with sample weights for XGBoost and Naive Bayes
        if model_name in ["XGBoost", "Naive Bayes"]:
            sample_weights = compute_sample_weight("balanced", y_train)
            model.fit(X_train, y_train, sample_weight=sample_weights)
        else:
            model.fit(X_train, y_train)        
            
        # Evaluate the model
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        accuracy = accuracy_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1_weighted = f1_score(y_test, y_pred, average='weighted')
        cohen_kappa = cohen_kappa_score(y_test, y_pred)
        positive_precision = precision_score(y_test, y_pred, pos_label=1)
        negative_precision = precision_score(y_test, y_pred, pos_label=0)
        error_rate = 1 - accuracy
        loss = log_loss(y_test, y_pred_proba)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        # Calculate the composite score based on these metrics
        composite_score_value = composite_score({
            "Accuracy": accuracy,
            "Recall": recall,
            "F1 Weighted": f1_weighted,
            "Cohen's Kappa": cohen_kappa,
            "Positive Precision": positive_precision,
            "Negative Precision": negative_precision,
            "Error Rate": error_rate,
            "Loss": loss,
            "roc_auc": roc_auc
        })
        # Compute metrics
        
        metrics_dict = {
            "Accuracy": round(accuracy,3),
            "Recall": round(recall,3),
            "F1 Weighted": round(f1_weighted,3),
            "Cohen's Kappa": round(cohen_kappa,3),
            "Positive Precision": round(positive_precision,3),
            "Negative Precision": round(negative_precision,3),
            "Error Rate": round(error_rate,3),
            "Loss": round(loss,3),
            "roc_auc": round(roc_auc,3),
            "Composite Score": round(composite_score_value,3)
        }
        
        # Store the scores
        for key, value in metrics_dict.items():
            scores[key].append(value)
            
    # Average the scores across all folds
    avg_scores = {key: round(np.mean(values),3) for key, values in scores.items()}
    return avg_scores
    

for model_name, param_list in ablation_params.items():
    for config in param_list:
        mean_fpr = np.linspace(0, 1, 100)
        tprs = []
        aucs = []        
        highest_composite_score = -1
        best_overall_config = None
        best_overall_metrics = None
        # Merge base configuration with the specific ablation configuration
        merged_config = {**base_configs[model_name], **config}
        model = None
        
        if model_name == "Naive Bayes":
            model = GaussianNB(**merged_config)
        elif model_name == "Logistic Regression":
            model = LogisticRegression(**merged_config)
        elif model_name == "XGBoost":
            model = XGBClassifier(**merged_config)
        elif model_name == "Random Forest":
            model = RandomForestClassifier(**merged_config)
        start_time = time.time()
        
        # Fit the model with sample weights for XGBoost and Naive Bayes
        if model_name in ["XGBoost", "Naive Bayes"]:
            sample_weights = compute_sample_weight("balanced", Y_train)
            model.fit(X_train, Y_train, sample_weight=sample_weights)
        else:
            model.fit(X_train, Y_train)
            
        # Predictions and Metrics Calculation
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        kappa = cohen_kappa_score(Y_test, y_pred)
        fpr, tpr, thresholds = roc_curve(Y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0  
        metrics_dict = {
                "Accuracy": round(accuracy_score(Y_test, y_pred),3),
                "Recall": round(recall_score(Y_test, y_pred),3),
                "F1 Weighted": round(f1_score(Y_test, y_pred, average='weighted'),3),
                "Cohen's Kappa": round(kappa,3),
                "Positive Precision": round(precision_score(Y_test, y_pred, pos_label=1),3),
                "Negative Precision": round(precision_score(Y_test, y_pred, pos_label=0),3),
                "Error Rate": round(1 - accuracy_score(Y_test, y_pred),3),
                "Loss": round(log_loss(Y_test, y_pred_proba),3),
                "Computing Time (s)": round(time.time() - start_time,3),
                "roc_auc": round(roc_auc,4)
        }
        current_composite_score = composite_score(metrics_dict)
        config_key = f"{model_name} - {config}"
        # Check if current composite score meets the threshold
        if current_composite_score > threshold:
            # Calculate ROC curve data
            fpr, tpr, _ = roc_curve(Y_test, y_pred_proba)
            roc_auc = auc(fpr, tpr)
            # Store ROC curve data in the dictionary
            roc_curve_data[config_key] = {
                'fpr': fpr.tolist(),  # Convert numpy array to list
                'tpr': tpr.tolist(),  # Convert numpy array to list
                'roc_auc': round(roc_auc,4)
            }             
        if current_composite_score > highest_composite_score:
            highest_composite_score = current_composite_score
            best_overall_config = f"{model_name} - {config}"
            best_overall_metrics = {
                "Accuracy": round(accuracy_score(Y_test, y_pred),3),
                "Recall": round(recall_score(Y_test, y_pred),3),
                "F1 Weighted": round(f1_score(Y_test, y_pred, average='weighted'),3),
                "Cohen's Kappa": round(kappa,3),
                "Positive Precision": round(precision_score(Y_test, y_pred, pos_label=1),3),
                "Negative Precision": round(precision_score(Y_test, y_pred, pos_label=0),3),
                "Error Rate": round(1 - accuracy_score(Y_test, y_pred),3),
                "Loss": round(log_loss(Y_test, y_pred_proba),3),
                "Computing Time (s)": round(time.time() - start_time,3),
                "roc_auc": round(roc_auc,3),
                "Composite score": round(current_composite_score,3),
            }
           
        # Update results for each configuration
        
        update_results(results_file_path, config_key, best_overall_metrics)
        print(f"Best overall configuration: {best_overall_config}")
        print("Metrics:", best_overall_metrics)

# Load results
results_df = pd.read_csv(results_file_path)


# Assuming the 'Configuration' column contains strings like "XGBoost - {'n_estimators': 100, 'max_depth': 3, ...}"
# We need to extract hyperparameters from this string

def extract_hyperparameters(config_str):
    try:
        # Extract the dictionary part of the string
        config_str = config_str.split(' - ')[1].strip()
        # Initialize an empty dictionary
        config_dict = {}
        # Use regex to find all key-value pairs in the string
        pattern = r"'(\w+)':\s*([^,}]+)"
        matches = re.findall(pattern, config_str)
        # Iterate over the matches and add them to the dictionary
        for key, value in matches:
            if '.' in value or 'e' in value:
                # Convert to float if the value is a decimal or in scientific notation
                config_dict[key] = float(value)
            else:
                # Convert to int if the value is an integer
                config_dict[key] = int(value)
        return config_dict
    except Exception as e:
        print(f"Error parsing config string: {config_str} | Error: {e}")
        return {}    
    

# Apply the function to each row
hyperparam_df = results_df['Configuration'].apply(extract_hyperparameters).apply(pd.Series)


hyperparam_columns = hyperparam_df.columns
for col in hyperparam_df.columns:
    results_df[col] = hyperparam_df[col]
    

metric_elements = results_df.columns
key_metric = metric_elements[10] 
    
# Calculate standard deviation of AUC-ROC for each hyperparameter
std_devs = {col: results_df.groupby(col)[key_metric].std() for col in hyperparam_columns}

# Drop columns from results_df that are already in hyperparam_df
results_df_filtered = results_df.drop(columns=hyperparam_df.columns)

# Concatenate hyperparam_df and the filtered results_df
combined_df = pd.concat([hyperparam_df, results_df_filtered], axis=1)


# Identify top hyperparameters based on standard deviation
top_hyperparameters = sorted(std_devs.items(), key=lambda x: -max(x[1]))
top_hyperparameters = [(param, max(std)) for param, std in top_hyperparameters]

# Display top hyperparameters
print("Top Impactful Hyperparameters (based on Std Dev of Composite score):")
for param, std_dev in top_hyperparameters:
    print(f"{param}: {std_dev}")


# Assuming hyperparam1 and hyperparam2 are your top two hyperparameters
# Assuming top_hyperparameters is a list of tuples like ('hyperparam_name', std_dev_value)
hyperparam1_name = top_hyperparameters[0][0]  # First hyperparameter name
hyperparam2_name = top_hyperparameters[1][0]  # Second hyperparameter name

# Group by the top two hyperparameters and get the max AUC-ROC score for each combination
agg_data = results_df.groupby([hyperparam1_name, hyperparam2_name])[metric_elements[10]].max().reset_index()
# Pivot the data for heatmap
# Correcting the pivot method call
heatmap_data = agg_data.pivot(index=hyperparam1_name, columns=hyperparam2_name, values=metric_elements[10])


plt.figure(figsize=(10, 8))
sns.heatmap(heatmap_data, annot=True, cmap='viridis', fmt=".3f")
plt.title(f"Heatmap of Composite score for {hyperparam1_name} and {hyperparam2_name}")
plt.xlabel(hyperparam2_name)
plt.ylabel(hyperparam1_name)
plt.savefig(fig_outfile_path)
plt.show()




