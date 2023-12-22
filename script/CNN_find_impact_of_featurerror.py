#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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



randomfeatures = [
'Difficulty memorizing lessons',
'Satisfied with living conditions',
'Long commute',
'Financial difficulties',
'Irregular rhythm of meals',
'Unbalanced meals',
'Eating junk food',
'Irregular rhythm or unbalanced meals',
'Physical activity(3 levels)',
'Physical activity(2 levels)',
'Prehypertension or hypertension',
'Cigarette smoker (5 levels)',
'Cigarette smoker (3 levels)',
'Drinker (3 levels)',
'Drinker (2 levels)',
'Binge drinking',
'Marijuana use']

def parse_arguments():
    parser = argparse.ArgumentParser(description="Run ML model for mental health prediction")
    parser.add_argument("--mode", choices=["depression", "anxiety"], required=True, help="Choose the mode: depression or anxiety")
    parser.add_argument("--type", choices=randomfeatures, required=True, help="Choose the feature to perturb")
    parser.add_argument('--kfold', type=int, default=10, help='Number of splits for K-Fold cross-validation (default: 10).')
    parser.add_argument('--epochs', type=int, default=60, help='Number of epochs for CNN training.')
    args = parser.parse_args()
    return args

CNNpath = "path/to/your/CNNmodel/"
yourpath = "path/to/your/"
file_path = os.path.join(CNNpath, "callback.txt")

X1 = pd.read_csv('path/to/your/original_processed_data.csv')
y1 = np.loadtxt('path/to/your/depression_status.csv', delimiter=',', dtype=int)
y2 = np.loadtxt('path/to/your/anxiety_status.csv', delimiter=',', dtype=int)



#========= Make pertubation data.    

selected_features = randomfeatures

def clean_feature_name(feature_name):
    # Replace spaces, parentheses, and brackets with underscores
    feature_name = re.sub(r"[\s()]", "_", feature_name)
    # Remove consecutive underscores
    feature_name = re.sub(r"_+", "_", feature_name)
    # Remove trailing underscores
    feature_name = feature_name.rstrip("_")
    return feature_name

if __name__ == "__main__":
    args = parse_arguments()
    feature_to_perturb = args.type
    index1 = selected_features.index(feature_to_perturb)
    # Depending on the mode, set the appropriate target variable
    if args.mode == "depression":
        y = y1
        mode_label = "depression"
        cname = f"d{index1}"
    elif args.mode == "anxiety":
        y = y2
        mode_label = "anxiety"
        cname = f"a{index1}"
    # Prepare the data for the selected feature
    
    num_kfold = args.kfold
    num_epochs = args.epochs 
    cleaned_feature = clean_feature_name(feature_to_perturb)


outfilename = = os.path.join(CNNpath, f"CNN_{cleaned_feature}_perturbation_{mode_label}_results.csv")

    
def perturb_feature(X, feature, prob_threshold, all_possible_values):
    perturbed_X = X.copy()
    random_mask = np.random.rand(*perturbed_X[feature].shape) < prob_threshold
    random_choices = np.random.choice(all_possible_values, size=perturbed_X[feature].shape)
    perturbed_X.loc[random_mask, feature] = random_choices[random_mask]
    return perturbed_X


def update_results(file_path, config, metrics):
    new_row = pd.DataFrame([{**{"Configuration": config}, **metrics}])
    with open(file_path, 'a') as f:
        new_row.to_csv(f, header=f.tell()==0, index=False)

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

            

desired_order = [
    "Accuracy", "Recall", "F1 Weighted", "Cohen's Kappa", "Positive Precision",
    "Negative Precision", "Error Rate", "Loss", "Computing Time (s)", "roc_auc",
    "Composite Score"
]

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
def cnn_train_and_evaluate(model, X_train, y_train, X_val, y_val, epochs=num_epochs, batch_size=16):
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


# Assuming df, y, and selected_features are defined
kf = KFold(n_splits=num_kfold, shuffle=True, random_state=42)

all_results = []

feature = feature_to_perturb
all_possible_values = X[feature].unique()
fold_results = []
    
for train_index, test_index in kf.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y[train_index], y[test_index]          
    oversample = SMOTE(k_neighbors=11)
    X_train, y_train = oversample.fit_resample(X_train, y_train)        
    # Fit the model on original data
    #cnn_model = get_cnn_model(X_train.shape[1]);
    #trained_model, epoch, original_metrics = cnn_train_and_evaluate(cnn_model, X_train, y_train, X_test, y_test)
    #original_metrics['Composite Score'] = composite_score(original_metrics)
    # Perturb the feature
    perturbed_X_train = perturb_feature(X_train, feature, prob_threshold=0.2, all_possible_values=all_possible_values)
    cnn_model_perturb = get_cnn_model(perturbed_X_train.shape[1]);
    perturbed_trained_model, epoch, perturbed_metrics = cnn_train_and_evaluate(cnn_model_perturb, perturbed_X_train, y_train, X_test, y_test)
    perturbed_metrics['Composite Score'] = composite_score(perturbed_metrics)

    fold_results.append({
        'fold': train_index.tolist(),
        #'original_metrics': original_metrics,
        'perturbed_metrics': perturbed_metrics
    })

# Calculate averages
#avg_original_metrics = {metric: round(np.mean([result['original_metrics'][metric] for result in fold_results]),3) for metric in original_metrics.keys()}
avg_perturbed_metrics = {metric: round(np.mean([result['perturbed_metrics'][metric] for result in fold_results]),3) for metric in perturbed_metrics.keys()}

all_results.append({
    'feature': feature,
    #'average_original_metrics': avg_original_metrics,
    'average_perturbed_metrics': avg_perturbed_metrics
})


        
# Save the results
results_df = pd.DataFrame(all_results)
results_df.to_csv(outfilename, index=False)

