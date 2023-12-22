## Evaluating Machine Learning Stability in Predicting Depression and Anxiety Amidst Subjective Response Errors

## Installation

git clone https://github.com/wailimku/MDD_GAD.git

cd MDD_GAD.git

pip install -r requirements.txt


## Data preprocessing script
data_preprocessing.py

## Ablation test scripts
1) CNN_ablationtest.py
2) Randomforest_ablation_test.py
3) xgboost_ablation_test.py
4) logregress_ablation_test.py
5) naivebayes_ablation_test.py

## Model comparison
####Description
This script is designed to evaluate machine learning models for predicting mental health conditions, specifically depression and anxiety. It allows users to perturb specific features in the dataset to assess the stability of the model predictions under varying conditions.

### Command-Line Arguments
--mode: Choose the mode for the script. Options are 'anxiety' or 'depression'.
--perturb: Choose the datatype for the script. Options are 'original' or 'perturbation'.
--kfold: Number of splits for K-Fold cross-validation. Default is 10.
--epochs: Number of epochs for CNN training. Default is 60.


### How to Run
Prepare Data: Place your datasets (original_processed_data.csv, perturbed_data_0.2.txt, depression_status.csv, and anxiety_status.csv) in the specified paths.
Run the Script: Use the command line to navigate to the script's directory and run it using Python. 
Example command:
python script_name.py --mode depression --type Difficulty_memorizing_lessons --kfold 10 --epochs 60

### Outputs
A CSV file containing the evaluation results of the model on the perturbed dataset. The file includes metrics like accuracy, recall, F1 score, and the custom composite score for each perturbed feature.

### Notes
Modify the file paths in the script according to your directory structure.


## Study of impacts of feature errors

### How to Run
Prepare Data: Place your datasets (original_processed_data.csv, depression_status.csv, and anxiety_status.csv) in the specified paths.
Run the Script: Use the command line to navigate to the script's directory and run it using Python. Example command:

python script_name.py --mode depression --type Difficulty_memorizing_lessons --kfold 10 --epochs 60

### Outputs
A CSV file containing the evaluation results of the model on the perturbed dataset. The file includes metrics like accuracy, recall, F1 score, and the custom composite score for each perturbed feature.

### Visualization
run the script "maketable.py" (Note: change input file)
