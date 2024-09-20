import os
import platform
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pylab import *
import jieba
from wordcloud import WordCloud, ImageColorGenerator, STOPWORDS
from PIL import Image


sns.set_palette("hls")

from matplotlib.font_manager import FontProperties
system = platform.system()

font = FontProperties(fname='fonts/opentype/noto/NotoSerifCJK-Black.ttc')


def get_all_scores(corpus):
    scores_dict = defaultdict(list)

    for obj in corpus:
        quality = obj["quality"]
        for aspect, result in quality.items():
            if result["score"] >= 0:
                scores_dict[aspect].append(result["score"])

    print("{:6s} | {:5s} | {:5s} | {:5s} | {:5s} | {} ".format("", "Mean", "Std", "Min", "Max", "Count"))
    for aspect, score_list in scores_dict.items():
        mean_score = np.mean(score_list)
        std_score = np.std(score_list)
        min_screo = min(score_list)
        max_score = max(score_list)
        print("{:5s} | {:5.2f} | {:5.2f} | {:5d} | {:5d} | {}".format(aspect, mean_score, std_score, min_screo, max_score, len(score_list)))

    return scores_dict


def get_wordcloud(corpus, text_column, figure_dir):
    text_list = [obj[text_column] for obj in corpus]
    text = "\n".join(text_list)

    wordlist = jieba.cut(text)
    wordlist = [w for w in wordlist if len(w) > 1]
    space_list = ' '.join(wordlist)

    backgroud = np.array(Image.open(Path(__file__).parent / "resources/HIT.jfif"))

    with open(Path(__file__).parent / "resources/hit_stopwords.txt", "r", encoding="utf-8") as f:
        stopwords = [w.rstrip() for w in f.readlines()]

    wc = WordCloud(width=1400, height=2200,
                   background_color='white',
                   mode='RGB',
                   mask=backgroud,
                   max_words=500,
                   stopwords=STOPWORDS.update(stopwords),
                   max_font_size=150,
                   relative_scaling=0.6,
                   random_state=50,
                   scale=2,
                   font_path=str(Path(__file__).parent / "resources/simsun.ttf"),
                   ).generate(space_list)

    image_color = ImageColorGenerator(backgroud)
    wc.recolor(color_func=image_color)

    plt.imshow(wc)
    plt.axis('off')
    plt.show()
    wc.to_file(os.path.join(figure_dir, "wordcloud.png"))


def get_plot(scores_dict: dict, figure_dir: str):
    sns.set_palette("hls")
    fig, axs = plt.subplots(1, 5, figsize=(25, 5))

    color_list = ["#FF55BB", "#00DFA2", "#FFD3A3", "#0079FF", "#F6FA70"]
    for i, (aspect, score_list) in enumerate(scores_dict.items()):
            
        sns.histplot(score_list, bins=10, color=color_list[i], ax=axs[i])
        sns.kdeplot(score_list, color="seagreen", lw=3, ax=axs[i])
        # sns.distplot(score_list, bins=10, kde_kws={"color": "seagreen", "lw": 3}, hist_kws={"color": color_list[i]}, ax=axs[i])
        axs[i].set_title(aspect, fontproperties=font)

    plt.savefig(os.path.join(figure_dir, "quality_hist.png"))
    plt.show()


def scores_visualization(corpus, text_column, figure_dir):
    get_wordcloud(corpus, text_column, figure_dir)
    scores_dict = get_all_scores(corpus)
    get_plot(scores_dict, figure_dir)