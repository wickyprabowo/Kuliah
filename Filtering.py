import math
import operator
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neighbors.classification import KNeighborsClassifier
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

df = pd.read_csv('dataset_sms_spam _v1.csv', skiprows=1, names=['Message', 'Status'])

#regex menghilangkan simbol, link, dan kata double
df_message = df["Message"].replace(r'http\S+', '', regex=True)\
    .replace(r'www\S+', '', regex=True)\
    .replace(r'tsel\S+', '', regex=True)\
    .replace(r'[-*#:]?([\d]+([^ ]?[a-zA-Z_/]*))+', '', regex=True)\
    .replace(r'\s([a-zA-Z])\1+\s', '', regex=True)
df_status = df["Status"]

#membagi message sesuai dengan kelasnya
df_message_0 = df_message[df["Status"] == 0]
df_status_0 = df_status[df["Status"] == 0]
message0_train, message0_test, status0_train, status0_test = train_test_split(df_message_0, df_status_0, test_size=0.2, random_state=4)

df_message_1 = df_message[df["Status"] == 1]
df_status_1 = df_status[df["Status"] == 1]
message1_train, message1_test, status1_train, status1_test = train_test_split(df_message_1, df_status_1, test_size=0.2, random_state=4)

df_message_2 = df_message[df["Status"] == 2]
df_status_2 = df_status[df["Status"] == 2]
message2_train, message2_test, status2_train, status2_test = train_test_split(df_message_2, df_status_2, test_size=0.2, random_state=4)

#menggabungkan 80% message dn 20% train
df_message_train = pd.concat([message0_train, message1_train, message2_train])
df_status_train = pd.concat([status0_train, status1_train, status2_train])

df_message_test = pd.concat([message0_test, message1_test, message2_test])
df_status_test = pd.concat([status0_test, status1_test, status2_test])

#StopWord
idn_stopWord = StopWordRemoverFactory().get_stop_words()

#Stemming
stemming = StemmerFactory().create_stemmer()
analyzer = CountVectorizer().build_analyzer()
def stemmed_words(doc):
    return (stemming.stem(w) for w in analyzer(doc))

cv = TfidfVectorizer(stop_words = idn_stopWord, analyzer=stemmed_words)

message_train_cv = cv.fit_transform(df_message_train)
getFitur = cv.get_feature_names()
a = message_train_cv.toarray()

#clasification
def cosineSimilarity (testData, trainingData):
    distance = 0
    penyebutCosineTest = 0
    penyebutCosineTrain = 0
    pembilangCosine = 0
    for i in range(len(getFitur)):
        pembilangCosine += (testData.item(i) * trainingData.item(i))
        penyebutCosineTest += pow(testData.item(i),2)
        penyebutCosineTrain += pow(trainingData.item(i),2)
    distance = 1 - (pembilangCosine/(math.sqrt(penyebutCosineTest) * math.sqrt(penyebutCosineTrain)))
    return distance

def kNearestNeighbor(setTestData, setTrainingData, k):
    distance = []
    neighbors = []
    #testData = 1
    for i in range(len(setTestData)):
        for j in range(len(setTrainingData)):
            dist = cosineSimilarity(setTestData[i], setTrainingData[j])
            distance.append((df_status_train.iloc[j], dist))
        distance.sort(key=operator.itemgetter(1))
        for l in range(k):
            neighbors.append(distance[l])
    if k == 1 :
        return neighbors[0][0]
    else:
        vote = {}
        for x in range(len(neighbors)):
            response = neighbors[x][0]
            if response in vote:
                vote[response] += 1
            else:
                vote[response] = 1
        sortedVotes = sorted(vote.items(), key=operator.itemgetter(1), reverse=True)
        return sortedVotes

new_message = ["aku sedang sakit hari ini"]
message_test_cv = cv.transform(new_message)
b = message_test_cv.toarray()
knn = kNearestNeighbor(b, a, 10)
print(knn)


'''clasification = KNeighborsClassifier(n_neighbors=3)
message_train_clasification = clasification.fit(message_train_cv, df_status_train)

accuracy = clasification.score(message_test_cv, df_status_test)

prediction = clasification.predict(message_test_cv)

print(df_message_test)
print(prediction)
print('\n', "DATA TRAINING : ")
print(df_message_train, '\n', df_status_train)
print('\n', "DATA TEST : ")
print(df_message_test, '\n', df_status_test)
print('\n', "FEATURE OF DATA TRAINING : ")
print(getFitur, '\n', len(getFitur))
print(idn_stopWord)'''