#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Standard library imports
import time

# Third-party imports for data processing and numeric operations
import numpy as np
import pandas as pd
import xarray as xr

# Visualization library
import matplotlib.pyplot as plt

# Machine learning and data preprocessing imports from scikit-learn
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score, roc_auc_score, roc_curve, auc
from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split, StratifiedShuffleSplit
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier

# Imports for advanced modeling and interpretation
from xgboost import XGBClassifier
import shap

# Data profiling and reporting
from ydata_profiling import ProfileReport

# IPython display utilities (if you're using Jupyter Notebooks)
from IPython.display import HTML


def load_and_initial_process(filepath):
    """
    Loads the dataset and performs initial processing.
    Args:
    - filepath: Path to the CSV file.

    Returns:
    - A pandas DataFrame with initial processing done.
    """
    df = pd.read_csv(filepath, index_col="id")
    # Any initial processing steps can be added here
    return df


def clean_data(df):
    """
    Cleans the provided DataFrame.
    Args:
    - df: pandas DataFrame to be cleaned.

    Returns:
    - Cleaned DataFrame.
    """
    # give the dtypes of the columns if the data was squeeky clean
    dtypes = {
    'id' : np.int32,
    'Age (4 levels)' : object,
    'Gender' : object,
    'French nationality' : object,
    'Field of study' : object,
    'Year of university' : object,
    'Learning disabilities' : object,
    'Difficulty memorizing lessons' : object,
    'Professional objective' : object,
    'Informed about opportunities' : object,
    'Satisfied with living conditions' : object,
    'Living with a partner/child' : object,
    'Parental home' : object,
    'Having only one parent' : object,
    'At least one parent unemployed' : object,
    'Siblings' : object,
    'Long commute' : object,
    'Mode of transportation' : object,
    'Financial difficulties' : object,
    'Grant' : object,
    'Additional income' : object,
    'Public health insurance ' : object,
    'Private health insurance ' : object,
    'C.M.U.' : object,
    'Irregular rhythm of meals' : object,
    'Unbalanced meals' : object,
    'Eating junk food' : object,
    'On a diet' : object,
    'Irregular rhythm or unbalanced meals' : object,
    'Physical activity(3 levels)' : object,
    'Physical activity(2 levels)' : object,
    'Weight (kg)' : np.int32,
    'Height (cm)' : np.int32,
    'Overweight and obesity' : object,
    'Systolic blood pressure (mmHg)' : np.int32,
    'Diastolic blood pressure (mmHg)' : np.int32,
    'Prehypertension or hypertension' : object,
    'Heart rate (bpm)' : np.int32,
    'Abnormal heart rate' : object,
    'Distant visual acuity of right eye (score /10)' : np.int32,
    'Distant visual acuity of left eye (score /10)' : np.int32,
    'Close visual acuity of right eye (score /10)' : np.int32,
    'Close visual acuity of left eye (score /10)' : np.int32,
    'Decreased in distant visual acuity' : object,
    'Decreased in close visual acuity' : object,
    'Urinalysis (glycosuria)' : object,
    'Urinalysis (proteinuria)' : object,
    'Urinalysis (hematuria)' : object,
    'Urinalysis leukocyturia)' : object,
    'Urinalysis (positive nitrite test)' : object,
    'Abnormal urinalysis' : object,
    'Vaccination up to date' : object,
    'Control examination needed' : object,
    'Anxiety symptoms' : object,
    'Panic attack symptoms' : object,
    'Depressive symptoms' : object,
    'Cigarette smoker (5 levels)' : object,
    'Cigarette smoker (3 levels)' : object,
    'Drinker (3 levels)' : object,
    'Drinker (2 levels)' : object,
    'Binge drinking' : object,
    'Marijuana use' : object,
    'Other recreational drugs' : object}
    
    constant_imputer = SimpleImputer(strategy="constant", fill_value = "miss")
    
    df['Age (4 levels)'] = df['Age (4 levels)'].astype("category")
    df['Gender'] = df['Gender'].astype("category")
    df['French nationality'] = df['French nationality'].astype("category")
    df['Field of study'] = df['Field of study'].astype("category")
    df['Year of university'] = df['Year of university'].astype("category")
    df['Learning disabilities'] = df['Learning disabilities'].astype("category")
    df['Difficulty memorizing lessons'] = df['Difficulty memorizing lessons'].astype("category")
    df['Professional objective'] = df['Professional objective'].astype("category")
    df['Informed about opportunities'] = df['Informed about opportunities'].astype("category")
    df['Satisfied with living conditions'] = df['Satisfied with living conditions'].astype("category")
    df['Living with a partner/child'] = df['Living with a partner/child'].astype("category")
    df['Parental home'] = df['Parental home'].astype("category")
    df['Having only one parent'] = df['Having only one parent'].astype("category")
    df['At least one parent unemployed'] = df['At least one parent unemployed'].astype("category")
    df['Siblings'] = df['Siblings'].astype("category")
    df['Long commute'] = df['Long commute'].astype("category")
    df['Mode of transportation'] = df['Mode of transportation'].astype("category")
    df['Financial difficulties'] = df['Financial difficulties'].astype("category")
    df['Grant'] = df['Grant'].astype("category")
    df['Additional income'] = df['Additional income'].astype("category")
    df['Public health insurance '] = df['Public health insurance '].astype("category")
    df['Private health insurance '] = df['Private health insurance '].astype("category")
    df['C.M.U.'] = df['C.M.U.'].astype("category")
    df['Irregular rhythm of meals'] = df['Irregular rhythm of meals'].astype("category")
    df['Unbalanced meals'] = df['Unbalanced meals'].astype("category")
    df['Eating junk food'] = df['Eating junk food'].astype("category")
    df['On a diet'] = df['On a diet'].astype("category")
    df['Irregular rhythm or unbalanced meals'] = df['Irregular rhythm or unbalanced meals'].astype("category")
    df['Physical activity(3 levels)'] = df['Physical activity(3 levels)'].astype("category")
    df['Physical activity(2 levels)'] = df['Physical activity(2 levels)'].astype("category")
    df['Overweight and obesity'] = df['Overweight and obesity'].astype("category")
    df['Prehypertension or hypertension'] = df['Prehypertension or hypertension'].astype("category")
    df['Abnormal heart rate'] = df['Abnormal heart rate'].astype("category")
    df['Decreased in distant visual acuity'] = df['Decreased in distant visual acuity'].astype("category")
    df['Decreased in close visual acuity'] = df['Decreased in close visual acuity'].astype("category")
    df['Urinalysis (glycosuria)'] = df['Urinalysis (glycosuria)'].astype("category")
    df['Urinalysis (proteinuria)'] = df['Urinalysis (proteinuria)'].astype("category")
    df['Urinalysis (hematuria)'] = df['Urinalysis (hematuria)'].astype("category")
    df['Urinalysis leukocyturia)'] = df['Urinalysis leukocyturia)'].astype("category")
    df['Urinalysis (positive nitrite test)'] = df['Urinalysis (positive nitrite test)'].astype("category")
    df['Abnormal urinalysis'] = df['Abnormal urinalysis'].astype("category")
    df['Vaccination up to date'] = df['Vaccination up to date'].astype("category")
    df['Control examination needed'] = df['Control examination needed'].astype("category")
    df['Anxiety symptoms'] = df['Anxiety symptoms'].astype("category")
    df['Panic attack symptoms'] = df['Panic attack symptoms'].astype("category")
    df['Depressive symptoms'] = df['Depressive symptoms'].astype("category")
    df['Cigarette smoker (5 levels)'] = df['Cigarette smoker (5 levels)'].astype("category")
    df['Cigarette smoker (3 levels)'] = df['Cigarette smoker (3 levels)'].astype("category")
    df['Drinker (3 levels)'] = df['Drinker (3 levels)'].astype("category")
    df['Drinker (2 levels)'] = df['Drinker (2 levels)'].astype("category")
    df['Binge drinking'] = df['Binge drinking'].astype("category")
    df['Marijuana use'] = df['Marijuana use'].astype("category")
    df['Other recreational drugs'] = df['Other recreational drugs'].astype("category")
    
    integer_columns = df.dtypes == "int64"
    numeric_columns = df.dtypes == "float64"
    category_columns = df.dtypes == "category"
    
    cat_mask_object = (df.dtypes == "category")
    cat_mask_object = df.columns[cat_mask_object].tolist()
    cat_mask_numeric = (df.dtypes == "float64")
    cat_mask_numeric = df.columns[cat_mask_numeric].tolist()
    cat_mask_integer = (df.dtypes == "int64")
    cat_mask_integer = df.columns[cat_mask_integer].tolist()
    
    numeric_columns_float64 = df[cat_mask_numeric].astype("float64").fillna(0)
    numeric_columns_int64 = df[cat_mask_integer].astype("int64").fillna(0)
    
    df[cat_mask_object] = constant_imputer.fit_transform(df[cat_mask_object])
    df_clean = pd.concat([numeric_columns_float64, numeric_columns_int64, df[cat_mask_object]], axis = 1)
    
    df_clean['Depressive symptoms'] = df_clean['Depressive symptoms'].replace("yes\t","yes")
    df_clean['Depressive symptoms'] = df_clean['Depressive symptoms'].replace("no\t","no")
    df_clean['Depressive symptoms'] = df_clean['Depressive symptoms'].replace("\tyes","yes")
    df_clean['Depressive symptoms'] = df_clean['Depressive symptoms'].replace("\tno","no")
    df_clean['Anxiety symptoms'] = df_clean['Anxiety symptoms'].replace("yes\t","yes")
    df_clean['Anxiety symptoms'] = df_clean['Anxiety symptoms'].replace("no\t","no")
    df_clean['Anxiety symptoms'] = df_clean['Anxiety symptoms'].replace("\tyes","yes")
    df_clean['Anxiety symptoms'] = df_clean['Anxiety symptoms'].replace("\tno","no")
    df_clean['Panic attack symptoms'] = df_clean['Panic attack symptoms'].replace("yes\t","yes")
    df_clean['Panic attack symptoms'] = df_clean['Panic attack symptoms'].replace("no\t","no")
    df_clean['Panic attack symptoms'] = df_clean['Panic attack symptoms'].replace("\tyes","yes")
    df_clean['Panic attack symptoms'] = df_clean['Panic attack symptoms'].replace("\tno","no")
    
    # subsetting columns with another boolean mask for categorical columns and object columns
    cat_mask_obj2 = (df_clean.dtypes == "object") | (df_clean.dtypes == "category")
    cat_mask_object2 = df_clean.columns[cat_mask_obj2].tolist()
    
    # remove the column classification 
    cat_mask_object2.remove('Depressive symptoms')
    cat_mask_object2.remove('Anxiety symptoms')
    cat_mask_object2.remove('Panic attack symptoms')
    
    df_clean2 = df_clean[cat_mask_object2]
    feature_name = list(df_clean2.columns.values.tolist())
    for i in feature_name:
        le = LabelEncoder()
        label = le.fit_transform(df_clean2[i])
        xlabel=np.unique(label, return_index=True)
        new_val = xlabel[0]
        rep_index = xlabel[1]+1
        old_val = df_clean2[i]	
        num_idx = rep_index.shape[0]
        for j in range(num_idx):
            df_clean2[i] = df_clean2[i].replace(old_val[rep_index[j]],new_val[j])
            
    concat_cols = np.hstack((df_clean2.values, df_clean[cat_mask_numeric].values,df_clean[cat_mask_integer].values))
    
    df_cat_var = pd.DataFrame(df_clean2, columns=feature_name,index=list(range(1,4185)))
    
    concat_cols_df = pd.concat([df_clean[cat_mask_integer],df_clean[cat_mask_numeric], df_cat_var], axis=1) 
    
    # now get the target variable into a numeric form
    col_preprocess1 = df_clean["Depressive symptoms"].replace("yes", 1) 
    final_col_preprocess1 = col_preprocess1.replace("no", 0)
    y1 = final_col_preprocess1.values
    
    col_preprocess2 = df_clean["Anxiety symptoms"].replace("yes", 1) 
    final_col_preprocess2 = col_preprocess2.replace("no", 0)
    y2 = final_col_preprocess2.values
    df_processed = concat_cols_df
    
    return df_processed, y1, y2

def randomize_features(df, features, prob=0.2):
    """
    Randomly alters some values in specified features of a dataframe.
    Args:
    - df: pandas DataFrame.
    - features: List of column names to be randomized.
    - prob: Probability of a value being altered.
    """
    np.random.seed(42)
    for feature in features:
        unique_values = pd.unique(df[feature])
        random_vector = np.random.rand(len(df), 1)
        swap_indices = np.where(random_vector < prob)[0]
        for idx in swap_indices:
            if idx in df.index:
                current_value = df.at[idx, feature]
                possible_values = [val for val in unique_values if val != current_value]
                df.at[idx, feature] = np.random.choice(possible_values)
        print(f'Randomized feature: {feature}')
    return df



def main():
# Load and initial process
    df = load_and_initial_process('path/to/your/raw_data.csv')
    
    # Clean data
    X, z1, z2 = clean_data(df)
    
    # Randomize features
    random_features = [
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
    df_random = randomize_features(X, random_features)
    
    X.to_csv('path/to/your/orginal_precessed_data.csv')
    np.savetxt('path/to/your/depression_status.csv', z1, delimiter=',', fmt='%i')
    np.savetxt('path/to/your/anxiety_status.csv', z2, delimiter=',', fmt='%i')
    df_random.to_csv('path/to/your/perturbed_data_0.2.txt')


if __name__ == "__main__":
    main()

