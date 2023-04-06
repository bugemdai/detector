import os
import glob
import core
import config
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


def forester():
    core.feature_required(config.PATH_DATASET, 'dataset')
    # required_features = core.feature_sort(config.PATH_DATASET, 'dataset')
    df = pd.read_csv(f'{config.PATH_DATASET}/dataset.csv')

    # TODO: переделать!!!!
    # df.drop(columns=df.columns[required_features[-1]], axis=1, inplace=True)
    # df.drop(columns=df.columns[required_features[-1]], axis=1, inplace=True)
    # df.drop(columns=df.columns[required_features[-1]], axis=1, inplace=True)
    # df.drop(columns=df.columns[required_features[-1]], axis=1, inplace=True)
    # df.drop(columns=df.columns[required_features[-1]], axis=1, inplace=True)

    labels = np.array(df.pop("labels"))
    train, test, train_labels, test_labels = train_test_split(df,
                                                              labels,
                                                              stratify=labels,
                                                              test_size=0.3,
                                                              random_state=50,)

    # pipe = Pipeline([("scaler", MinMaxScaler()), ("rfc", RandomForestClassifier())])

    # TODO: Подобрать правельные параметры
    # param_grid = {'rfc__max_features': ['auto', 'sqrt', 'log2'],
    #               'rfc__n_estimators': np.arange(1, 250, 20),
    #               'rfc__max_depth': [3, 5, 8, 12, 15], }
    # grid = GridSearchCV(pipe, param_grid=param_grid, n_jobs=-2, cv=5)
    # grid.fit(np.array(train), np.array(train_labels))
    #
    # print("Best learn rfc: {:.2f}".format(grid.best_score_))
    # print("Best test rfc: {:.2f}".format(grid.score(np.array(test), np.array(test_labels))))
    # print("Best parameters rfc: {}".format(grid.best_params_))


def data_consolidation():
    all_filenames = [i for i in glob.glob(f'{config.PATH_CSV}/*.csv')]
    consolidate_df = pd.concat([pd.read_csv(file) for file in all_filenames])
    return consolidate_df


def smile_to_csv():
    for filename in os.listdir(config.PATH_AUDIO):
        if filename.endswith(".wav"):
            file = os.path.join(f'{config.PATH_AUDIO}/{filename}')
            smile_value_df = core.smile_create(file)
            smile_value_df.drop(smile_value_df.tail(1).index, inplace=True)

            if filename.endswith("T.wav"):
                smile_value_df.insert(smile_value_df.shape[1], 'labels', 1)
            else:
                smile_value_df.insert(smile_value_df.shape[1], 'labels', 0)

            smile_value_df.to_csv(f'{config.PATH_CSV}/{filename[:-4]}.csv', index=False)
        else:
            continue


def start_job():
    smile_to_csv()
    df = data_consolidation()
    df.to_csv(f'{config.PATH_DATASET}/dataset.csv', index=False)


if __name__ == '__main__':
    # start_job()
    forester()
