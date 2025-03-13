import os
import glob
import core
import config
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier


def forester():
    """
    Function to perform feature selection and train a Random Forest Classifier.
    """
    # Ensure required features are present
    core.feature_required(config.PATH_DATASET, 'dataset')
    
    # Load dataset
    df = pd.read_csv(f'{config.PATH_DATASET}/dataset.csv')

    # Extract labels
    labels = np.array(df.pop("labels"))
    
    # Split the data into training and testing sets
    train, test, train_labels, test_labels = train_test_split(
        df, labels, stratify=labels, test_size=0.3, random_state=50
    )

    # Define a pipeline with a scaler and a Random Forest Classifier
    pipe = Pipeline([
        ("scaler", MinMaxScaler()),
        ("rfc", RandomForestClassifier())
    ])

    # Define parameter grid for GridSearch
    param_grid = {
        'rfc__max_features': ['auto', 'sqrt', 'log2'],
        'rfc__n_estimators': np.arange(1, 250, 20),
        'rfc__max_depth': [3, 5, 8, 12, 15],
    }

    # Perform GridSearchCV
    grid = GridSearchCV(pipe, param_grid=param_grid, n_jobs=-2, cv=5)
    grid.fit(np.array(train), np.array(train_labels))

    # Print best scores and parameters
    print("Best learn rfc: {:.2f}".format(grid.best_score_))
    print("Best test rfc: {:.2f}".format(grid.score(np.array(test), np.array(test_labels))))
    print("Best parameters rfc: {}".format(grid.best_params_))


def data_consolidation():
    """
    Function to consolidate all CSV files in the audio path into a single DataFrame.
    """
    all_files = glob.glob(os.path.join(config.PATH_AUDIO, "*.wav"))
    consolidate_df = pd.concat([pd.read_csv(file) for file in all_files])
    return consolidate_df


def smile_to_csv():
    """
    Convert audio files in the specified directory to CSV format.
    """
    for filename in os.listdir(config.PATH_AUDIO):
        if filename.endswith(".wav"):
            # Read the audio file and convert to CSV
            smile_df = pd.read_csv(f'{config.PATH_AUDIO}/{filename}')
            smile_df['labels'] = 0
            smile_to_csv_path = f'{config.PATH_CSV}/{filename[:-4]}.csv'
            smile_df.to_csv(smile_to_csv_path, index=False)
        else:
            continue


def start_job():
    smile_to_csv()
    df = data_consolidation()
    df.to_csv(f'{config.PATH_DATASET}/dataset.csv', index=False)


if __name__ == '__main__':
    forester()
