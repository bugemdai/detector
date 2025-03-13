import utils
import audiofile
import opensmile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import SelectKBest


def smile_create(file):
    signal, sampling_rate = audiofile.read(file, duration=25, )
    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.emobase,
        feature_level=opensmile.FeatureLevel.LowLevelDescriptors,
    )
    smile_value_df = smile.process_signal(signal, sampling_rate)
    return smile_value_df


def feature_required(in_path, filename):
    df = pd.read_csv(f'{in_path}/{filename}.csv',
                     delimiter=',',
                     header=None,
                     index_col=False)

    data = df.iloc[1:, :-1]
    labels_data = df.iloc[1:, -1]
    header_names = df.iloc[0, :-1]

    normed_data = utils.min_max_scaler(data)

    f_class_best_features = SelectKBest(f_classif, k='all')
    f_class_best_features.fit(normed_data, labels_data)
    f_class_evaluation = -np.log10(f_class_best_features.pvalues_)
    f_class_evaluation_scores = f_class_evaluation[f_class_evaluation[:].argsort()[::-1]]

    f_class_indices = np.arange(data.shape[-1])
    utils.plt_figure_show(f_class_indices, f_class_evaluation_scores)
    utils.plt_figure_header_show(f_class_evaluation, f_class_evaluation_scores, header_names)

    chi2_best_features = SelectKBest(chi2, k='all')
    chi2_best_features.fit(normed_data, labels_data)
    chi2_evaluation = -np.log10(chi2_best_features.pvalues_)
    chi2_evaluation_scores = chi2_evaluation[chi2_evaluation[:].argsort()[::-1]]

    chi2_best_indices = np.arange(data.shape[-1])
    utils.plt_figure_show(chi2_best_indices, chi2_evaluation_scores)
    utils.plt_figure_header_show(chi2_evaluation, chi2_evaluation_scores, header_names)
