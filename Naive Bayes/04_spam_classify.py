import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import accuracy_score

email_list = [
    {'email title':'free game only today','spam':True},
    {'email title':'cheapest flight deal','spam':True},
    {'email title':'limited time offer only today only today','spam':True},
    {'email title':'today meeting schedule','spam':False},
    {'email title':'your flight schedule attached','spam':False},
    {'email title':'your credit card statement','spam':False}
]

df = pd.DataFrame(email_list)
df['label'] = df['spam'].map({True:1,False:0})

df_x = df['email title']
df_y = df['label']

cv = CountVectorizer(binary = True)
x_traincv = cv.fit_transform(df_x)

encoded_input = x_traincv.toarray()

bnb = BernoulliNB()
y_train = df_y.astype('int')
bnb.fit(x_traincv,y_train)


test_email_list = [
    {'email title': 'free flight offer', 'spam': True},
    {'email title': 'hey traveler free flight deal', 'spam': True},
    {'email title': 'limited free game offer', 'spam': True},
    {'email title': 'today flight schedule', 'spam': False},
    {'email title': 'your credit card attached', 'spam': False},
    {'email title': 'free credit card offer only today', 'spam': False}
]

test_df = pd.DataFrame(test_email_list)
test_df['label'] = test_df['spam'].map({True:1,False:0})
test_x = test_df['email title']
test_y = test_df['label']

x_testcv = cv.transform(test_x)

predictions = bnb.predict(x_testcv)
print(accuracy_score(test_y,predictions))