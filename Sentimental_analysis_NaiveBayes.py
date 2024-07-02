import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import ComplementNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import CategoricalNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import re

nltk.download('wordnet')

def preprocess_text(text):
    # Remove special characters and <br>
    text = re.sub(r'<br />', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Tokenization
    tokens = word_tokenize(text)
    # Removing stopwords
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in filtered_tokens]
    return " ".join(lemmatized_tokens)


gnb=GaussianNB()
com=ComplementNB()
ber=BernoulliNB()
cat=CategoricalNB()
mul=MultinomialNB()

df=pd.read_csv("H:\\sem 6\\ml lab\\mc_movie.csv")
print(df.head())

df3=df.drop('textID',axis=1)
df3['text'] = df3['text'].apply(preprocess_text)

x=df3.iloc[:,0]
df3.replace(['positive','negative','neutral'],[1,-1,0],inplace=True)
y=df3['sentiment']

x=CountVectorizer().fit_transform(df3['text'])

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

#multinomial NB
x_test_re=x_test.reshape(-1,1)
y_pred=mul.fit(x_train,y_train).predict(x_test)
print('accuracy:',accuracy_score(y_pred,y_test))
print(classification_report(y_pred,y_test))
cm = confusion_matrix(y_test, y_pred)
import seaborn as sns
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

#complement NB
y_pred=com.fit(x_train,y_train).predict(x_test)
print('accuracy:',accuracy_score(y_pred,y_test))
print(classification_report(y_pred,y_test))
cm = confusion_matrix(y_test, y_pred)
import seaborn as sns
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

#Bernoulli
y_pred=ber.fit(x_train,y_train).predict(x_test)
print('accuracy:',accuracy_score(y_pred,y_test))
print(classification_report(y_pred,y_test))
cm = confusion_matrix(y_test, y_pred)
import seaborn as sns
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()




