#!/usr/bin/env python
# coding: utf-8

# In[1]:


import time
import os
import json
import argparse
import itertools
from collections import OrderedDict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import torch
import xgboost as xgb
from imblearn.over_sampling import SMOTE, SVMSMOTE
from keras.callbacks import Callback
from scipy import interp
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, auc, classification_report, cohen_kappa_score, confusion_matrix,
    f1_score, log_loss, make_scorer, precision_recall_fscore_support, precision_score,
    recall_score, roc_auc_score, roc_curve
)
from sklearn.model_selection import (
    GridSearchCV, KFold, RepeatedStratifiedKFold, StratifiedShuffleSplit, train_test_split, cross_val_score
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_sample_weight
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.layers import Conv1D, Dense, Dropout, Flatten, MaxPooling1D
from tensorflow.keras.metrics import AUC
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.optimizers import Adam
from torch.nn.parallel import DistributedDataParallel as DDP
from xgboost import XGBClassifier
import shap
from ydata_profiling import ProfileReport

# If you're using Jupyter Notebooks
from IPython.display import HTML


# Set up argument parsing
parser = argparse.ArgumentParser(description="Process well-being data.")
parser.add_argument('--mode', type=str, choices=['anxiety', 'depression'], required=True, help='Choose the mode for the script.')
parser.add_argument('--perturb', type=str, choices=['original', 'perturbation'], required=True, help='Choose the datatype for the script.')
parser.add_argument('--kfold', type=int, default=10, help='Number of splits for K-Fold cross-validation (default: 10).')
parser.add_argument('--epochs', type=int, default=60, help='Number of epochs for CNN training.')

args = parser.parse_args()

CNNpath = "path/to/your/CNNmodel/"
yourpath = "path/to/your/"
file_path = os.path.join(CNNpath, "callback.txt")

X1 = pd.read_csv('path/to/your/original_processed_data.csv')
y1 = np.loadtxt('path/to/your/depression_status.csv', delimiter=',', dtype=int)
X2 = pd.read_csv('path/to/your/perturbed_data_0.2.txt')
y2 = np.loadtxt('path/to/your/anxiety_status.csv', delimiter=',', dtype=int)



if args.perturb == "original":
    # Load the original dataset
    X = X1
elif args.perturb == 'perturbation':
    # Load the perturbation dataset
     = X2

# Choose label based on condition
if args.mode == "anxiety":
    y3 = y2
    output_suffix = "anxiety"
elif args.mode == 'depression':
    y3 = y1
    output_suffix = "depression"


roc_curve_output_path = os.path.join(yourpath, f'ROC_curve_{output_suffix}_{args.perturb}_data.pdf')
comparison_results_output_path = os.path.join(yourpath, f'Five_model_comparison_results_{output_suffix}_{args.perturb}_data.csv')

    
if isinstance(X, pd.DataFrame):
    X = X.values
    

def composite_score(row, weight_auc=2, weight_kappa=0.25):
    recall = row['Recall']
    cs = weight_auc * row['roc_auc'] + weight_kappa * row['Cohen\'s Kappa']
    return cs * (recall > 0)


class CompositeScoreCallback(Callback):
    def __init__(self, validation_data, file_path):
        super(CompositeScoreCallback, self).__init__()
        self.validation_data = validation_data
        self.best_weights = None
        self.best_score = -np.Inf
        self.file_path = file_path
        
    def on_epoch_end(self, epoch, logs=None):
        X_val, y_val = self.validation_data
        y_pred = self.model.predict(X_val).ravel()
        y_pred_binary = (y_pred > 0.5).astype(int)
        
        # Calculate the metrics you want to include in your composite score
        recall = recall_score(y_val, y_pred_binary)
        cohen_kappa = cohen_kappa_score(y_val, y_pred_binary)
        roc_auc = roc_auc_score(y_val, y_pred)
        
        # Define your composite score formula
        cs = 2 * roc_auc + 0.25 * cohen_kappa
        composite_score = cs * (recall > 0)
        
        # Check if the composite score is improved
        if composite_score > self.best_score:
            self.best_score = composite_score
            self.best_weights = self.model.get_weights()
            # Save the best model
            self.model.save(self.file_path)
            print(f"\nEpoch {epoch + 1}: CompositeScore improved to {composite_score:.4f}, saving model to {self.file_path}")
        else:
            print(f"\nEpoch {epoch + 1}: CompositeScore did not improve from {self.best_score:.4f}")

def get_cnn_model(input_dim=59, num_filters=64, kernel_size=3, pool_size=2, use_dropout=True, dropout_rate=0.5, num_dense_neurons=50, num_conv_layers=1):
    model = Sequential()
    for i in range(num_conv_layers):
        if i == 0:
            model.add(Conv1D(filters=num_filters, kernel_size=kernel_size, activation='relu', input_shape=(input_dim, 1)))
        else:
            model.add(Conv1D(filters=num_filters, kernel_size=kernel_size, activation='relu'))
            model.add(MaxPooling1D(pool_size=pool_size))
        
        if use_dropout:
            model.add(Dropout(dropout_rate))
            
    model.add(Flatten())
    model.add(Dense(num_dense_neurons, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    optimizer = Adam(learning_rate=0.0005)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model


# Training and evaluation function
def cnn_train_and_evaluate(model, X_train, y_train, X_val, y_val, epochs=args.epochs, batch_size=16):
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, min_lr=0.00001, verbose=1)
    composite_score_callback = CompositeScoreCallback(validation_data=(X_val, y_val), file_path=file_path)
    callbacks_list = [composite_score_callback, reduce_lr]
     
    best_metrics = {}
    best_epoch = -1
    best_model_weights = None
    start_time = time.time() 
    
    for epoch in range(epochs):
        model.fit(X_train, y_train, batch_size=batch_size, epochs=20, validation_data=(X_val, y_val), callbacks=callbacks_list, verbose=1)
            # After training, set the best weights back to the model
        if composite_score_callback.best_weights is not None:
            model.set_weights(composite_score_callback.best_weights)
    
        val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=0)
        
        y_pred = model.predict(X_val).ravel()
        y_pred_binary = (y_pred > 0.5).astype(int)
        fpr, tpr, _ = roc_curve(y_val, y_pred)
        roc_auc = auc(fpr, tpr)
        
        metrics = {
            "Accuracy": round(accuracy_score(y_val, y_pred_binary),3),
            "Recall": round(recall_score(y_val, y_pred_binary),3),
            "F1 Weighted": round(f1_score(y_val, y_pred_binary, average='weighted'),3),
            "Cohen's Kappa": round(cohen_kappa_score(y_val, y_pred_binary),3),
            "Positive Precision": round(precision_score(y_val, y_pred_binary, pos_label=1),3),
            "Negative Precision": round(precision_score(y_val, y_pred_binary, pos_label=0),3),
            "Error Rate": round(1 - val_accuracy,3),
            "Loss": round(val_loss,3),
            "roc_auc": round(roc_auc,3)
        }
        comp_score = composite_score(metrics)
        metrics['Composite Score'] = comp_score  
        
        print(f"Epoch {epoch + 1}/{epochs} - ", " - ".join(f"{k}: {v:.4f}" for k, v in metrics.items()))
        
        if best_epoch == -1 or comp_score > (best_metrics['Composite Score'] if best_metrics else 0):
            best_epoch = epoch
            best_metrics = metrics
            best_model_weights = model.get_weights() 
            
    training_time = time.time() - start_time  # Total training time
    best_metrics["Computing Time (s)"] = round(training_time, 3)
    best_overall_metrics = OrderedDict((key, best_metrics[key]) for key in desired_order)
    
    #model.set_weights(best_model_weights)
    return model, best_epoch, best_overall_metrics


def plot_roc_curves(model_results):
    plt.figure()
    for model_name, data in model_results.items():
        plt.plot(data['mean_fpr'], data['mean_tpr'], label=f'{model_name} (AUC = {data["mean_auc"]:.2f} ± {data["std_auc"]:.2f})')
        
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    #plt.show()
    

base_configs = {
    "Random Forest": {"class_weight": "balanced", "n_estimators": 1000,"n_jobs": 18,"max_depth": 12, "min_samples_split": 12,"min_samples_leaf": 10},
    "XGBoost": {"objective": 'binary:logistic', 'n_estimators': 50, 'max_depth': 7, 'learning_rate': 0.0001, 'gamma': 0, 'alpha': 2.5, 'colsample_bytree': 0.2, "n_jobs": 18},
    "Logistic Regression": {"max_iter": 100, 'penalty': 'l2', 'C': 10, 'fit_intercept': False, 'class_weight': 'balanced', 'solver': 'newton-cg', 'l1_ratio': None},
    "Naive Bayes": {"var_smoothing": 0.0001}}

desired_order = [
    "Accuracy", "Recall", "F1 Weighted", "Cohen's Kappa", "Positive Precision",
    "Negative Precision", "Error Rate", "Loss", "Computing Time (s)", "roc_auc",
    "Composite Score"
]



def evaluate_model(model_name, X, y, n_splits = args.kfold):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    mean_fpr = np.linspace(0, 1, 100)
    tprs = []
    aucs = []
    model_results = {}
    # Define the keys for the metrics you expect to calculate
    metrics_keys = ["Accuracy", "Recall", "F1 Weighted", "Cohen's Kappa", "Positive Precision", "Negative Precision", "Error Rate", "Loss", "Computing Time (s)","roc_auc","Composite Score"]
    results = {metric: [] for metric in metrics_keys}
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = y[train_index], y[test_index]      
        # Apply SMOTE only if the model is a CNN
        if model_name == "CNN":
            oversample = SMOTE(k_neighbors=11)
            X_train, Y_train = oversample.fit_resample(X_train, Y_train)
            cnn_model = get_cnn_model(X_train.shape[1]);
            trained_model, epoch, best_overall_metrics = cnn_train_and_evaluate(cnn_model, X_train, Y_train, X_test, Y_test)
            y_pred_proba = trained_model.predict(X_test).ravel()
            fpr, tpr, thresholds = roc_curve(Y_test, y_pred_proba)
            roc_auc = auc(fpr, tpr)
            aucs.append(roc_auc)
            tprs.append(np.interp(mean_fpr, fpr, tpr))
            tprs[-1][0] = 0.0   
        else:
            highest_composite_score = -1
            best_overall_config = None
            best_overall_metrics = None
            # Merge base configuration with the specific ablation configuration
            merged_config = {**base_configs[model_name]}
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
            tprs.append(np.interp(mean_fpr, fpr, tpr))
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
            metrics_dict["Composite Score"] = current_composite_score            
            if current_composite_score > highest_composite_score:
                highest_composite_score = current_composite_score
                fpr, tpr, thresholds = roc_curve(Y_test, y_pred_proba)
                roc_auc = auc(fpr, tpr)
                aucs.append(roc_auc)
                tprs.append(np.interp(mean_fpr, fpr, tpr))
                tprs[-1][0] = 0.0               
                best_overall_config = f"{model_name}"
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
                    "Composite Score": round(current_composite_score,3),
                }
        for key, value in best_overall_metrics.items():
            results[key].append(value)
            
    results_mean = {key: round(np.mean(values),3) for key, values in results.items()}
    results_std = {key: round(np.std(values),3) for key, values in results.items()}
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)    
    model_results[model_name] = {
            'mean_fpr': mean_fpr, 
            'mean_tpr': mean_tpr, 
            'mean_auc': mean_auc, 
            'std_auc': std_auc
            }
    
    return results_mean, results_std, model_results

# Define your models
model_items = ["XGBoost", "Random Forest", "Logistic Regression", "Naive Bayes", "CNN"]
# Define your metrics_keys
metrics_keys = ["Accuracy", "Recall", "F1 Weighted", "Cohen's Kappa", "Positive Precision", "Negative Precision", "Error Rate", "Loss", "Computing Time (s)","roc_auc","Composite Score"]

# Dictionary to hold the aggregated results
aggregated_results = {model: {'mean': {}, 'std': {}} for model in model_items}

# ... similar functions or instances for other 4 models

model_results2 = {}
for models in model_items:
    results_mean, results_std, model_results = evaluate_model(models, X, y3)
    model_results2[models] = model_results
    aggregated_results[models]['mean'] = results_mean
    aggregated_results[models]['std'] = results_std    


    
adjusted_model_results = {model: data[model] for model, data in model_results2.items()}
plot_roc_curves(adjusted_model_results)
plt.savefig(roc_curve_output_path)


# Preparing the DataFrame
data = {'Model': model_items}
for key in metrics_keys:
    data[f'Mean {key}'] = [aggregated_results[model]['mean'].get(key, None) for model in model_items]
    data[f'Std {key}'] = [aggregated_results[model]['std'].get(key, None) for model in model_items]

# Creating and 
df = pd.DataFrame(data)
df.to_csv(comparison_results_output_path, index=False)


# In[ ]:





# In[ ]:




