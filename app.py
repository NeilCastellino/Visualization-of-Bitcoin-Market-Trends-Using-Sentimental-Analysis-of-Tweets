import numpy as np
import sys
from flask import Flask, request, jsonify, render_template, redirect, url_for
from PIL import Image
from wordcloud import WordCloud, STOPWORDS
import os, shutil
import random
import pandas as pd
import csv
import json

app = Flask(__name__)
static_folder_path = os.path.join('static', 'images')
app.config['UPLOAD_FOLDER'] = static_folder_path

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/piechart_textblob')
def piechart_textblob():
    with open('static/data/sentiment_analysis.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        positive_count = 0
        negative_count = 0
        neutral_count = 0
        row_count=0
        for row in csv_reader:
            row_count+=1
            if row_count==1:
                continue

            if row[5]=="positive":
                positive_count+=1
            elif row[5]=="negative":
                negative_count+=1
            elif row[5]=="neutral":
                neutral_count+=1
    return render_template("piechart_textblob.html", positive_count = positive_count, neutral_count=neutral_count, negative_count=negative_count)

@app.route('/piechart_vader')
def piechart_vader():
    with open('static/data/sentiment_analysis.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        positive_count = 0
        negative_count = 0
        neutral_count = 0
        row_count=0
        for row in csv_reader:
            row_count+=1
            if row_count==1:
                continue

            if row[7]=="positive":
                positive_count+=1
            elif row[7]=="negative":
                negative_count+=1
            elif row[7]=="neutral":
                neutral_count+=1
    return render_template("piechart_vader.html", positive_count = positive_count, neutral_count=neutral_count, negative_count=negative_count)

@app.route('/bargraph_textblob')
def bargraph_textblob():
    with open('static/data/sentiment_analysis.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        positive_count = 0
        negative_count = 0
        neutral_count = 0
        row_count=0
        for row in csv_reader:
            row_count+=1
            if row_count==1:
                continue

            if row[5]=="positive":
                positive_count+=1
            elif row[5]=="negative":
                negative_count+=1
            elif row[5]=="neutral":
                neutral_count+=1
    return render_template("bargraph_textblob.html", positive_count = positive_count, neutral_count=neutral_count, negative_count=negative_count)

@app.route('/bargraph_vader')
def bargraph_vader():
    with open('static/data/sentiment_analysis.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        positive_count = 0
        negative_count = 0
        neutral_count = 0
        row_count=0
        for row in csv_reader:
            row_count+=1
            if row_count==1:
                continue

            if row[7]=="positive":
                positive_count+=1
            elif row[7]=="negative":
                negative_count+=1
            elif row[7]=="neutral":
                neutral_count+=1
    return render_template("bargraph_vader.html", positive_count = positive_count, neutral_count=neutral_count, negative_count=negative_count)

@app.route('/scatterplot_btc_price_sentiment')
def scatterplot_btc_price_sentiment():
    with open('static/data/ml_data.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        btc_price_diff = []
        sentiment = []
        row_count=0
        for row in csv_reader:
            row_count+=1
            if row_count==1:
                continue
            btc_price_diff.append((float(row[5]) - float(row[6])))
            sentiment.append(float(row[2]))
    return render_template("scatterplot_btc_price_sentiment.html", btc_price_diff = btc_price_diff, sentiment = sentiment)

@app.route('/scatterplot_parquet')
def scatterplot_parquet():
    with open('static/data/parquet.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        polarity = []
        subjectivity = []
        row_count=0
        for row in csv_reader:
            row_count+=1
            if row_count==1:
                continue
            polarity.append(float(row[0]))
            subjectivity.append(float(row[1]))
    return render_template("scatterplot_parquet.html", polarity = polarity, subjectivity = subjectivity)

@app.route('/wordcloud')
def wordcloud():
    count = int(100)
    width = int(800)
    height = int(800)
    minfontsize = int(4)
    maxfontsize = int(150)
    background = np.array(Image.open("cloud.png"))
    stopwords = set(STOPWORDS)

    with open('static/data/sentiment_analysis.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        sentiment_words = ""
        row_count=0
        for row in csv_reader:
            row_count+=1
            if row_count==1:
                continue
            sentiment_words += str(row[3])
            sentiment_words+=" "

    wc = WordCloud(background_color="white",
                max_words=row_count,
                width = width,
                min_font_size = minfontsize,
                max_font_size = maxfontsize,
                height = height,
                mask = background,
                stopwords = stopwords)
    wc.generate(sentiment_words)

    random_num = random.randrange(0,1000,1)
    full_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'wordcloud'+str(random_num)+'.png')

    for filename in os.listdir(app.config['UPLOAD_FOLDER']):
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

    wc.to_file(full_filename)
    return render_template("wordcloud.html", card_image = full_filename)

@app.route('/wordcloud_positive')
def wordcloud_positive():
    count = int(100)
    width = int(800)
    height = int(800)
    minfontsize = int(4)
    maxfontsize = int(150)
    background = np.array(Image.open("cloud.png"))
    stopwords = set(STOPWORDS)

    with open('static/data/sentiment_analysis.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        sentiment_words = ""
        row_count=0
        for row in csv_reader:
            row_count+=1
            if row_count==1:
                continue

            if str(row[5]) == "positive":
                sentiment_words += str(row[3])
                sentiment_words+=" "
            else:
                row_count-=1

    wc = WordCloud(background_color="white",
                max_words=row_count,
                width = width,
                min_font_size = minfontsize,
                max_font_size = maxfontsize,
                height = height,
                mask = background,
                stopwords = stopwords)
    wc.generate(sentiment_words)

    random_num = random.randrange(0,1000,1)
    full_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'wordcloud'+str(random_num)+'.png')

    for filename in os.listdir(app.config['UPLOAD_FOLDER']):
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

    wc.to_file(full_filename)
    return render_template("wordcloud_positive.html", card_image = full_filename)

@app.route('/wordcloud_negative')
def wordcloud_negative():
    count = int(100)
    width = int(800)
    height = int(800)
    minfontsize = int(4)
    maxfontsize = int(150)
    background = np.array(Image.open("cloud.png"))
    stopwords = set(STOPWORDS)

    with open('static/data/sentiment_analysis.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        sentiment_words = ""
        row_count=0
        for row in csv_reader:
            row_count+=1
            if row_count==1:
                continue

            if str(row[5]) == "negative":
                sentiment_words += str(row[3])
                sentiment_words+=" "
            else:
                row_count-=1

    wc = WordCloud(background_color="white",
                max_words=row_count,
                width = width,
                min_font_size = minfontsize,
                max_font_size = maxfontsize,
                height = height,
                mask = background,
                stopwords = stopwords)
    wc.generate(sentiment_words)

    random_num = random.randrange(0,1000,1)
    full_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'wordcloud'+str(random_num)+'.png')

    for filename in os.listdir(app.config['UPLOAD_FOLDER']):
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

    wc.to_file(full_filename)
    return render_template("wordcloud_negative.html", card_image = full_filename)

@app.route('/wordcloud_neutral')
def wordcloud_neutral():
    count = int(100)
    width = int(800)
    height = int(800)
    minfontsize = int(4)
    maxfontsize = int(150)
    background = np.array(Image.open("cloud.png"))
    stopwords = set(STOPWORDS)

    with open('static/data/sentiment_analysis.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        sentiment_words = ""
        row_count=0
        for row in csv_reader:
            row_count+=1
            if row_count==1:
                continue

            if str(row[5]) == "neutral":
                sentiment_words += str(row[3])
                sentiment_words+=" "
            else:
                row_count-=1

    wc = WordCloud(background_color="white",
                max_words=row_count,
                width = width,
                min_font_size = minfontsize,
                max_font_size = maxfontsize,
                height = height,
                mask = background,
                stopwords = stopwords)
    wc.generate(sentiment_words)

    random_num = random.randrange(0,1000,1)
    full_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'wordcloud'+str(random_num)+'.png')

    for filename in os.listdir(app.config['UPLOAD_FOLDER']):
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

    wc.to_file(full_filename)
    return render_template("wordcloud_neutral.html", card_image = full_filename)

@app.route('/contact')
def contact():
    return render_template('contact.html')

if __name__ == "__main__":
    app.run(debug=True)