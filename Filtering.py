import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

df = pd.read_csv('dataset_sms_spam _v1.csv', skiprows=1, names=['Message', 'Status'])
idn_stopWord = StopWordRemoverFactory().get_stop_words()

df_message = df["Message"].replace(r'http\S+', '', regex=True)\
    .replace(r'www\S+', '', regex=True)\
    .replace(r'tsel\S+', '', regex=True)\
    .replace(r'[-*#:]?([\d]+([^ ]?[a-zA-Z_/]*))+', '', regex=True)\
    .replace(r'\s([a-cA-C])\1', '', regex=True)

df_status = df["Status"]
#reLink = ('((www.)[0-9a-z\./_+\(\)\$\#\&\!\?]+)')
#reNumber = '[-*#:]?([\d]+([^ ]?[a-zA-Z/]*))+'
reWord = '[a-z]+'

message_train, message_test, status_train, status_test = train_test_split(df_message, df_status, test_size=0.2, random_state=4)

cv = CountVectorizer(stop_words = idn_stopWord)

message_traincv = cv.fit_transform(message_train)
getFitur = cv.get_feature_names()
a = message_traincv.toarray()

#print(getFitur)
#print(len(getFitur))
#print(idn_stopWord)
#print(len(idn_stopWord))
#print(cv.inverse_transform(message_test))
#print(status_train)
#print(df_message_split_getValue)






