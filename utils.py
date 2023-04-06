import os
import moviepy.editor as mp
import numpy as np
import seaborn as sns

from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler


def plt_figure_header_show(evaluation, evaluation_scores, header_names):
    temp_overlap_df = []
    for t in range(len(evaluation_scores)):
        overlap = np.where(evaluation == evaluation_scores[t])
        temp_overlap_df.append(overlap[0][0])

    plt.figure(figsize=(10, 5))
    plt.clf()
    index_list = []
    label_list = []
    print(temp_overlap_df)
    for i in temp_overlap_df:
        sns.set_theme(style="darkgrid")
        df = sns.load_dataset(header_names[i], evaluation[i])
        sns.displot(
            df, x="flipper_length_mm", col="species", row="sex",
            binwidth=3, height=3, facet_kws=dict(margin_titles=True),
        )
        # plt.bar(header_names[i], evaluation[i])
        # index_list.append(header_names[i])
        # label_list.append(evaluation[i])
        # plt.title("Feature univariate")
        # plt.xlabel("Feature number")
        # plt.ylabel(r"Univariate score ($-Log(p_{value})$)")
        # plt.xticks(rotation=90)
        # plt.subplots_adjust(bottom=0.37)
    plt.show()


def min_max_scaler(df):
    scaler = MinMaxScaler()
    normalize_df = scaler.fit_transform(df)
    return normalize_df


def plt_figure_show(x, height):
    plt.figure(1)
    plt.clf()
    plt.bar(x, height)
    plt.title("Feature univariate score")
    plt.xlabel("Feature number")
    plt.ylabel(r"Univariate score ($-Log(p_{value})$)")
    plt.show()


def wmv_to_wav(in_path, out_path):
    for filename in os.listdir(in_path):
        if filename.endswith(".wmv"):
            video_clip = mp.VideoFileClip(f'{out_path}/{filename}')
            video_clip.audio.write_audiofile(f'{out_path}/{filename}.wav')
        else:
            continue


def counter(pd, in_path, filename, column):
    counter_df = pd.read_csv(f'{in_path}/{filename}.csv')
    counter_df = counter_df[f'{column}'].value_counts().rename_axis(f'{column}').reset_index(name='Amount')
    print(counter_df)
