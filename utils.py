import os
import moviepy.editor as mp
import numpy as np
import seaborn as sns

from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler


def plt_figure_header_show(evaluation, evaluation_scores, header_names):
    """
    Display a series of distribution plots for the given evaluation scores.

    Parameters:
    - evaluation: List of evaluation metrics.
    - evaluation_scores: List of scores to compare against evaluation metrics.
    - header_names: List of dataset names to load for plotting.
    """
    temp_overlap_df = []
    for t in range(len(evaluation_scores)):
        overlap = np.where(evaluation == evaluation_scores[t])
        temp_overlap_df.append(overlap[0][0])

    plt.figure(figsize=(10, 5))
    plt.clf()
    index_list = []
    label_list = []
    print(temp_overlap_df)  # Debugging: print the overlap indices
    for i in temp_overlap_df:
        sns.set_theme(style="darkgrid")
        df = sns.load_dataset(header_names[i], evaluation[i])
        sns.displot(
            df, x="flipper_length_mm", col="species", row="sex",
            binwidth=3, height=3, facet_kws=dict(margin_titles=True),
        )
    plt.show()


def min_max_scaler(df):
    """
    Normalize the dataframe using Min-Max scaling.

    Parameters:
    - df: DataFrame to be normalized.

    Returns:
    - Normalized DataFrame.
    """
    scaler = MinMaxScaler()
    normalize_df = scaler.fit_transform(df)
    return normalize_df


def plt_figure_show(x, height):
    """
    Display a bar plot for the given data.

    Parameters:
    - x: Labels for the x-axis.
    - height: Heights of the bars.
    """
    plt.figure(1)
    plt.clf()
    plt.bar(x, height)
    plt.title("Feature univariate score")
    plt.xlabel("Feature number")
    plt.ylabel(r"Univariate score ($-Log(p_{value})$)")
    plt.show()


def wmv_to_wav(in_path, out_path):
    """
    Convert WMV files to WAV audio files.

    Parameters:
    - in_path: Directory containing WMV files.
    - out_path: Directory to save the converted WAV files.
    """
    for filename in os.listdir(in_path):
        if filename.endswith(".wmv"):
            video_clip = mp.VideoFileClip(f'{out_path}/{filename}')
            video_clip.audio.write_audiofile(f'{out_path}/{filename}.wav')


def counter(pd, in_path, filename, column):
    """
    Count occurrences of values in a specified column of a CSV file.

    Parameters:
    - pd: Pandas module.
    - in_path: Directory containing the CSV file.
    - filename: Name of the CSV file (without extension).
    - column: Column name to count values from.
    """
    counter_df = pd.read_csv(f'{in_path}/{filename}.csv')
    counter_df = counter_df[f'{column}'].value_counts().rename_axis(f'{column}').reset_index(name='Amount')
    print(counter_df)
