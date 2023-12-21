#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import json
import time
import numpy as np
import pandas as pd
import xarray as xr
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, auc, classification_report, cohen_kappa_score, 
                             confusion_matrix, f1_score, log_loss, precision_recall_fscore_support, 
                             precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import (GridSearchCV, KFold, RepeatedStratifiedKFold, 
                                     StratifiedShuffleSplit, train_test_split, cross_val_score)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_sample_weight
from imblearn.over_sampling import SMOTE, SVMSMOTE
import matplotlib.pyplot as plt
from ydata_profiling import ProfileReport
import xgboost as xgb
from xgboost import XGBClassifier
import shap
import tensorflow as tf
from tensorflow.keras.callbacks import Callback, ReduceLROnPlateau
from tensorflow.keras.layers import Conv1D, Dense, Dropout, Flatten, MaxPooling1D
from tensorflow.keras.metrics import AUC
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.optimizers import Adam
import torch
import torch.distributed as dist
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, TensorDataset
from tabulate import tabulate

CNNpath = "path/to/your/CNNmodel/"
X = pd.read_csv('path/to/your/original_processed_data.csv')
y1 = np.loadtxt('path/to/your/depression_status_test.csv', delimiter=',', dtype=int)
file_path = os.path.join(CNNpath, "callback.txt")
results_file_path = os.path.join(CNNpath, "CNN_ablation_study_results_v1.csv")
best_config_file_path = os.path.join(CNNpath, "CNN_ablation_best_config_v1.csv")

X = X.values
y = y1

oversample = SMOTE(k_neighbors=11)
X_train, X_temp, y_train, y_temp= train_test_split(X, y1,test_size= 0.1, random_state=0)
X_train_over, y_train_over = oversample.fit_resample(X_train, y_train)

def update_results(file_path, config, metrics):
    # Define the desired order of your metrics
    desired_order = [
        'Accuracy',
        'Recall',
        'F1 Weighted',
        'Cohen\'s Kappa',
        'Positive Precision',
        'Negative Precision',
        'Error Rate',
        'Loss',
        'Computing Time (s)',
        'roc_auc',
        'Composite Score'  # Ensure this is the last metric if that's the desired position
    ]  
    # Combine config and metrics into one dictionary, ensuring metrics are ordered correctly
    combined_results = {**{"Configuration": str(config)}, **{k: metrics[k] for k in desired_order}}
    # Convert the combined results to a DataFrame
    results_df = pd.DataFrame([combined_results], columns=combined_results.keys())
    if os.path.exists(file_path):
        # Append new result
        results_df.to_csv(file_path, mode='a', header=False, index=False)
    else:
        # Create a new file with new result
        results_df.to_csv(file_path, header=True, index=False)

        
def composite_score(metrics, weight_auc=2, weight_kappa=0.25):
    recall = metrics['Recall']
    cs = weight_auc * metrics['roc_auc'] + weight_kappa * metrics['Cohen\'s Kappa']
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


            
# Define CNN model building function
def build_cnn_model(num_filters=64, kernel_size=3, pool_size=2, use_dropout=True, dropout_rate=0.5, num_dense_neurons=50, num_conv_layers=2):
    model = Sequential()
    for i in range(num_conv_layers):
        if i == 0:
            model.add(Conv1D(filters=num_filters, kernel_size=kernel_size, activation='relu', input_shape=(X_train.shape[1], 1)))
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
def train_and_evaluate(model, X_train, y_train, X_val, y_val, batch_size, epochs):
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, min_lr=0.00001, verbose=1)
    composite_score_callback = CompositeScoreCallback(validation_data=(X_val, y_val), file_path=file_path)
    callbacks_list = [composite_score_callback, reduce_lr]
    
    best_metrics = None
    best_epoch = -1
    best_model_weights = None
    start_time = time.time() 
    
    # Callback to reduce learning rate when a metric has stopped improving
    
    for epoch in range(epochs):
        model.fit(X_train, y_train, batch_size=batch_size, epochs=40, validation_data=(X_val, y_val), callbacks=callbacks_list, verbose=1)
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
    #model.set_weights(best_model_weights)
    return model, best_epoch, best_metrics



cnn_configs = [
    {
        "epochs": ep,
        "num_conv_layers": ncl,
        "num_dense_neurons": ndn,
        "num_filters": nf,
        "batch_size": bs
        
    }
    for ep in [20, 40, 60]  # Example epochs
    for ncl in [1, 2, 3]
    for ndn in [50, 75, 100]
    for nf in [32, 64]
    for bs in [8, 16, 32]  # Example batch sizes
]


results = []
# File paths and other initialization
        
for config in cnn_configs:
    config_without_epochs_batch_size = {k: v for k, v in config.items() if k not in ['epochs', 'batch_size']}
    # Now build the model without the epochs and batch_size in the config
    cnn_model = build_cnn_model(**config_without_epochs_batch_size)
    cnn_model = nn.DataParallel(cnn_model)
    cnn_model.to('cuda')
    
    trained_model, best_epoch, best_metrics= train_and_evaluate(cnn_model, X_train_over, y_train_over, X_temp, y_temp, config['batch_size'], config['epochs'])
    
    # Save model
    model_save_path = os.path.join(CNNpath, f"cnn_model_{config['num_conv_layers']}layers_{config['num_dense_neurons']}neurons.h5")
    trained_model.save(model_save_path)
    print(f"Best model for config {config} saved to {model_save_path}")
    # Record results and update file
    result = best_metrics  # best_metrics already contains the performance metrics
    result["Configuration"] = config  # Add the configuration to the results
    result["Best Epoch"] = best_epoch  # Add the best epoch to the results
    update_results(results_file_path, config, result)  # Pass the config and result dictionaries
    print(result)
        
# Update or append results



import pandas as pd
import matplotlib.pyplot as plt


# Read the CSV file
data = pd.read_csv(results_file_path)



data_rounded = data.round(2)
data_rounded.drop(columns=['best_epoch','Negative Precision', 'Computing Time (s)'], inplace=True)

plt.figure(figsize=(14, 8))  # Increase the figure size if needed
ax = plt.subplot(111, frame_on=False)
ax.xaxis.set_visible(False)
ax.yaxis.set_visible(False)

col_widths = [0.4, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]  # Example widths

table = plt.table(cellText=data_rounded.values, 
                  colLabels=data_rounded.columns, 
                  cellLoc='center', 
                  loc='center',
                  colWidths=col_widths)

table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 1)
#plt.show()
resultfig_save_path = os.path.join(CNNpath, 'cnn_ablation_study_results.pdf')
plt.savefig(results_save_path, bbox_inches='tight')
