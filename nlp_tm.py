import pandas as pd
import numpy as np

metin = """
A Scandal in Bohemia! 01
The Red-headed League,2
A Case, of Identity 33
The Boscombe Valley Mystery4
The Five Orange Pips1
The Man with? the Twisted Lip
The Adventure of the Blue Carbuncle
The Adventure of the Speckled Band
The Adventure of the Engineer's Thumb
The Adventure of the Noble Bachelor
The Adventure of the Beryl Coronet
The Adventure of the Copper Beeches"""

metin
# string'i df/array/seriye cevirmek

metin.split("\n")

v_metin = metin.split("\n")

v = pd.Series(v_metin)
v
metin_vektoru = v[1:len(v)]
metin_vektoru

mdf = pd.DataFrame(metin_vektoru,columns=["hikayeler"])
mdf

#buyuk-kucuk harf

d_mdf = mdf.copy()
d_mdf

d_mdf = d_mdf["hikayeler"].apply(lambda x: " ".join(x.lower() for x in x.split()))
d_mdf

#noktalama isaretlerinin silinmesi
type(d_mdf)

d_mdf = pd.DataFrame(d_mdf, columns=["hikayeler"])

#d_mdf = d_mdf.str.replace("[^\w\s]","")

d_mdf = d_mdf.replace("[^\w\s]","")

d_mdf

#sayıların silinmesi

#d_mdf = d_mdf.str.replace("\d","")

d_mdf = d_mdf.replace("\d","")
d_mdf
#stopwords

d_mdf = pd.DataFrame(d_mdf, columns=["hikayeler"])
import nltk
#nltk.download("stopwords")


from nltk.corpus import stopwords

sw = stopwords.words("english")
sw


d_mdf = d_mdf["hikayeler"].apply(lambda x: " ".join(x for x in x.split() if x not in sw))
d_mdf

# az geçen kelimelerin silinmesi
d_mdf = pd.DataFrame(d_mdf, columns = ["hikayeler"])

pd.Series(" ".join(d_mdf["hikayeler"]).split()).value_counts()

silinecek = ["01","33","2","4","1","identity","redheaded","pips"]

d_mdf = d_mdf["hikayeler"].apply(lambda x: " ".join(x for x in x.split() if x not in silinecek))
d_mdf

# Tokenization - cumleye kelimlere ayırmak,yeni bir vektor, liste olusturmaktır.
#nltk.download("punkt")
import textblob

from textblob import TextBlob

d_mdf = pd.DataFrame(d_mdf, columns = ["hikayeler"])
TextBlob(d_mdf["hikayeler"][1]).words

d_mdf["hikayeler"].apply(lambda x: TextBlob(x).words)

# Stemming - Bir kelimeyi köklerine indirgemeyi ifade eder.

from nltk.stem import PorterStemmer
st = PorterStemmer()

d_mdf["hikayeler"].apply(lambda x: " ".join([st.stem(i) for i in x.split()]))

# Lemmatization - Bir kelimeyi köklerine indirger stemming gibi ama kelimeyi daha iyi korur

from textblob import Word
nltk.download("wordnet")

d_mdf["hikayeler"].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))

#karsılastırma
mdf["hikayeler"][0:5]

d_mdf["hikayeler"][0:5]

## NPL UYGULAMALARI

# N-GRAM -- kelime kombinasyonlarını ifade etmek için kullanılır.ngrams degeri 3 olursa uc kelimeyi liste olarak gösterir

a = """Bu örneği anlaşılabilmesi için daha uzun bir metin üzerinden göstereceğim.
N-gram'lar birlikte kullanılan kelimelerin kombinasyolarını gösterir"""

TextBlob(a).ngrams(1)

# Part of speech tagging(POS) - metinde bulunan kelimelerin sıfat,zarf oldugunu tespit etmek icin kullanılır
nltk.download("averaged_perceptron_tagger")

TextBlob(d_mdf["hikayeler"][2]).tags

d_mdf["hikayeler"].apply(lambda x: TextBlob(x).tags)

# Chunking (shallow parsing) - Çıkarılan sıfat zarf durumlarını diyagram ile gösterebiliriz
pos = d_mdf["hikayeler"].apply(lambda x: TextBlob(x).tags)
pos

cumle = "R and Python are useful data science tools for the new or old data scientists who eager to do efficent data science task"

pos = TextBlob(cumle).tags
pos

reg_exp = "NP: {<DT>?<JJ>*<NN>}"
rp = nltk.RegexpParser(reg_exp)

sonuclar = rp.parse(pos)
sonuclar

print(sonuclar)
sonuclar.draw()

# Named Entity Recognition - kelimenin hangi nesne oldugunu belirtir. ör new york icin sehir der.
from nltk import word_tokenize, pos_tag, ne_chunk
nltk.download("maxent_ne_chunker")
nltk.download("words")

cumle = "Hadley is creative people who work for R Studio AND he attented conference at Newyork last year"
print(ne_chunk(pos_tag(word_tokenize(cumle))))

# MATEMATİKSEL İŞLEMLER VE BASİT ÖZELLİK ÇIKARIMI
# Harf/ Karakter Sayısı
o_df = d_mdf.copy()
o_df["hikayeler"]
o_df["hikayeler"].str.len()

o_df["harf_Sayisi"] = o_df["hikayeler"].str.len()

# Kelime Sayısı
a = "scandal in a bohemia"
a.split()
len(a.split())

o_df.iloc[0:1,0:1]

o_df["kelime_sayisi"] = o_df["hikayeler"].apply(lambda x: len(str(x).split(" ")))
o_df

# Özel Karakteri Yakalamak & Saydırmak

o_df["hikayeler"].apply(lambda x: len([x for x in x.split() if x.startswith("adventure")]))

o_df["ozel_karakter_sayisi"] = o_df["hikayeler"].apply(lambda x: len([x for x in x.split() if x.startswith("adventure")]))
o_df

# Sayıları Saklamak & Saydırmak

mdf["hikayeler"].apply(lambda x: len([x for x in x.split() if x.isdigit()]))

o_df["sayi_sayisi"] = mdf["hikayeler"].apply(lambda x: len([x for x in x.split() if x.isdigit()]))

o_df

# METIN GORSELLESTIRME
import pandas as pd
data = pd.read_csv("train.tsv",sep="\t")

data.head()
data.info()

# buyuk-kucuk donusumu
data["Phrase"] = data["Phrase"].apply(lambda x: " ".join(x.lower() for x in x.split()))

#noktalama isaretleri
data["Phrase"] = data["Phrase"].str.replace('[^\w\s]','')

#sayılar
data["Phrase"] = data["Phrase"].str.replace("\d","")

#stopwords
import nltk
nltk.download("stopwords")
from nltk.corpus import stopwords
sw = stopwords.words("english")
data["Phrase"] = data["Phrase"].apply(lambda x: " ".join(x for x in x.split() if x not in sw))

#seyreklerin silinmesi
sil = pd.Series("".join(data["Phrase"]).split()).value_counts()[-1000:]
data["Phrase"] = data["Phrase"].apply(lambda x:" ".join(x for x in x.split() if x not in sw))

#lemmi
from textblob import Word
#nltk.download('wordnet')
data['Phrase'] = data['Phrase'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()])) 

data['Phrase'].head(10)

#Terim Frekansı
tf1 = (data["Phrase"][1:10000]).apply(lambda x: pd.value_counts(x.split(" "))).sum(axis = 0).reset_index()

tf1.columns = ["words","tf"]
tf1.head()
tf1.info()
tf1.nunique()

#bar plot
a = tf1[tf1["tf"] > 1000]
a.plot.bar(x="words",y="tf");

# Wordcloud
import numpy as np
import pandas as pd
from os import path
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib.pyplot as plt

text = data["Phrase"][0]
wordcloud = WordCloud().generate(text)
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()


wordcloud = WordCloud(max_font_size=50,
                      max_words=100,
                      background_color="white").generate(text)
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()

wordcloud.to_file("kelime_bulutu.png")

#tum metin
text = " ".join(i for i in data.Phrase)
text

wordcloud = WordCloud(max_font_size=50,
                      background_color="white").generate(text)
plt.figure(figsize= [10,10])
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()

# Şablonlara Göre Word Cloud
vbo_mask = np.array(Image.open("tr.png"))
vbo_mask


wc = WordCloud(background_color="white",
               max_words=1000,
               mask = vbo_mask,
               contour_width = 3,
               contour_color="firebrick")

wc.generate(text)
wc.to_file("new.png")

plt.figure(figsize= [10,10])
plt.imshow(wc, interpolation="bilinear")
plt.axis("off")
plt.show()

# SENTIMENT ANALIZI VE SINIFLANDIRMA MODELLERI

from textblob import TextBlob
from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import decomposition, ensemble

import pandas, xgboost, numpy, textblob, string
from keras.preprocessing import sequence
from keras import layers, models, optimizers

import pandas as pd
data=pd.read_csv("train.tsv",sep="\t")

data.head()

data["Sentiment"].replace(0,value="negatif",inplace=True)
data["Sentiment"].replace(1,value="negatif",inplace=True)

data["Sentiment"].replace(3,value="pozitif",inplace=True)
data["Sentiment"].replace(4,value="pozitif",inplace=True)

data.head()

data = data[(data.Sentiment == "negatif") | (data.Sentiment == "pozitif")]

data.head()
data.groupby("Sentiment").count()

df= pd.DataFrame()
df["text"] = data["Phrase"]
df["label"] = data["Sentiment"]

df.head()

# METIN ON ISLEME
# buyuk-kucuk donusumu
df["text"] = df["text"].apply(lambda x: " ".join(x.lower() for x in x.split()))
#noktalama isaretleri
df["text"] = df["text"].str.replace('[^\w\s]','')
#sayılar
df["text"] = df["text"].str.replace("\d","")
#stopwords
import nltk
nltk.download("stopwords")
from nltk.corpus import stopwords
sw = stopwords.words("english")
df["text"] = df["text"].apply(lambda x: " ".join(x for x in x.split() if x not in sw))
#seyreklerin silinmesi
sil = pd.Series("".join(df["text"]).split()).value_counts()[-1000:]
df["text"] = df["text"].apply(lambda x:" ".join(x for x in x.split() if x not in sw))
#lemmi
from textblob import Word
#nltk.download('wordnet')
df['text'] = df['text'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()])) 
df['text'].head(10)


# Test- Train
train_x, test_x, train_y, test_y = model_selection.train_test_split(df["text"], df["label"],random_state=1)

encoder = preprocessing.LabelEncoder()

train_y = encoder.fit_transform(train_y)
test_y = encoder.fit_transform(test_y)

train_y[0:5]
test_y[0:5]

# Count Vectors
vectorizer = CountVectorizer()
vectorizer.fit(train_x)

x_train_count = vectorizer.transform(train_x)
x_test_count = vectorizer.transform(test_x)

vectorizer.get_feature_names_out()[0:5]
x_train_count.toarray()

# TF - IDF
#word level
tf_idf_word_vectorizer = TfidfVectorizer()
tf_idf_word_vectorizer.fit(train_x)

x_train_tf_idf_word = tf_idf_word_vectorizer.transform(train_x)
x_test_tf_idf_word = tf_idf_word_vectorizer.transform(test_x)

tf_idf_word_vectorizer.get_feature_names_out()[0:5]

x_train_tf_idf_word.toarray()

# N-gram level tf-idf
tf_idf_ngram_vectorizer = TfidfVectorizer(ngram_range=(2,3))
tf_idf_ngram_vectorizer.fit(train_x)

x_train_tf_idf_ngram = tf_idf_ngram_vectorizer.transform(train_x)
x_test_tf_idf_ngram = tf_idf_ngram_vectorizer.transform(test_x)

# characters level tf-idf
tf_idf_chars_vectorizer = TfidfVectorizer(analyzer="char",ngram_range=(2,3))
tf_idf_chars_vectorizer.fit(train_x)

x_train_tf_idf_chars = tf_idf_chars_vectorizer.transform(train_x)
x_test_tf_idf_chars = tf_idf_chars_vectorizer.transform(test_x)

# MAKINE OGRENMESİ İLE SENTIMENT SINIFLANDIRMASI
# Lojistik Regresyon
import warnings
warnings.filterwarnings("ignore")

loj =linear_model.LogisticRegression()
loj_model = loj.fit(x_train_count,train_y)
accuracy = model_selection.cross_val_score(loj_model, x_test_count,test_y,cv=10).mean()
print("Count Vectors Doğruluk Oranı:",accuracy)


loj =linear_model.LogisticRegression()
loj_model = loj.fit(x_train_tf_idf_word,train_y)
accuracy = model_selection.cross_val_score(loj_model, x_test_tf_idf_word,test_y,cv=10).mean()
print("Word-Level TF-IDF Doğruluk Oranı:",accuracy)

loj =linear_model.LogisticRegression()
loj_model = loj.fit(x_train_tf_idf_ngram,train_y)
accuracy = model_selection.cross_val_score(loj_model, x_test_tf_idf_ngram,test_y,cv=10).mean()
print("N-GRAM TF-IDF Doğruluk Oranı:",accuracy)

loj =linear_model.LogisticRegression()
loj_model = loj.fit(x_train_tf_idf_chars,train_y)
accuracy = model_selection.cross_val_score(loj_model, x_test_tf_idf_chars,test_y,cv=10).mean()
print("CHARLEVEL Doğruluk Oranı:",accuracy)

#Naive Bayes
nb = naive_bayes.MultinomialNB()
nb_model = nb.fit(x_train_count, train_y)
accuracy = model_selection.cross_val_score(nb_model, x_test_count,test_y,cv=10).mean()
print("Count Vectors Doğruluk Oranı:",accuracy)

nb = naive_bayes.MultinomialNB()
nb_model = nb.fit(x_train_tf_idf_word, train_y)
accuracy = model_selection.cross_val_score(nb_model, x_test_tf_idf_word,test_y,cv=10).mean()
print("Word-Level TF-IDF Doğruluk Oranı:",accuracy)

nb = naive_bayes.MultinomialNB()
nb_model = nb.fit(x_train_tf_idf_ngram, train_y)
accuracy = model_selection.cross_val_score(nb_model, x_test_tf_idf_ngram,test_y,cv=10).mean()
print("N-GRAM TF-IDF Doğruluk Oranı:",accuracy)

nb = naive_bayes.MultinomialNB()
nb_model = nb.fit(x_train_tf_idf_chars, train_y)
accuracy = model_selection.cross_val_score(nb_model, x_test_tf_idf_chars,test_y,cv=10).mean()
print("CHARLEVEL Doğruluk Oranı:",accuracy)

#Random Forests
rf = ensemble.RandomForestClassifier()
rf_model = rf.fit(x_train_count, train_y)
accuracy = model_selection.cross_val_score(rf_model, x_test_count,test_y,cv=10).mean()
print("Count Vectors Doğruluk Oranı:",accuracy)

rf = ensemble.RandomForestClassifier()
rf_model = rf.fit(x_train_tf_idf_word, train_y)
accuracy = model_selection.cross_val_score(rf_model, x_test_tf_idf_word,test_y,cv=10).mean()
print("Word-Level TF-IDF Doğruluk Oranı:",accuracy)

rf = ensemble.RandomForestClassifier()
rf_model = rf.fit(x_train_tf_idf_ngram, train_y)
accuracy = model_selection.cross_val_score(rf_model, x_test_tf_idf_ngram,test_y,cv=10).mean()
print("N-GRAM TF-IDF Doğruluk Oranı:",accuracy)

rf = ensemble.RandomForestClassifier()
rf_model = rf.fit(x_train_tf_idf_chars, train_y)
accuracy = model_selection.cross_val_score(rf_model, x_test_tf_idf_chars,test_y,cv=10).mean()
print("CHARLEVEL Doğruluk Oranı:",accuracy)

# XGBoost
xgb = xgboost.XGBClassifier()
xgb_model = xgb.fit(x_train_count,train_y)
accuracy = model_selection.cross_val_score(xgb_model, x_test_count,test_y,cv=10).mean()
print("Count Vectors Doğruluk Oranı:",accuracy)

xgb = xgboost.XGBClassifier()
xgb_model = xgb.fit(x_train_tf_idf_word,train_y)
accuracy = model_selection.cross_val_score(xgb_model, x_test_tf_idf_word,test_y,cv=10).mean()
print("Word-Level TF-IDF Doğruluk Oranı:",accuracy)

xgb = xgboost.XGBClassifier()
xgb_model = xgb.fit(x_train_tf_idf_ngram,train_y)
accuracy = model_selection.cross_val_score(xgb_model, x_test_tf_idf_ngram,test_y,cv=10).mean()
print("N-GRAM TF-IDF Doğruluk Oranı:",accuracy)

xgb = xgboost.XGBClassifier()
xgb_model = xgb.fit(x_train_tf_idf_chars,train_y)
accuracy = model_selection.cross_val_score(xgb_model, x_test_tf_idf_chars,test_y,cv=10).mean()
print("CHARLEVEL Doğruluk Oranı:",accuracy)


#loj_model.predict("yes i like this film") # hata verir

# yeni_yorum = pd.Series("this film is very nice and good i like it")

# v = CountVectorizer()
# v.fit(train_x)
# yeni_yorum = v.transform(yeni_yorum)

# loj_model.predict(yeni_yorum)

















