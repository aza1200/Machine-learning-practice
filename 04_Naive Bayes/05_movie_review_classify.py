import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

review_list = [
    {'movie_review': 'this is great great movie. I will watch again' , 'type': 'positive' },
    {'movie_review': 'I like this movie' , 'type': 'positive' },
    {'movie_review': 'amazing movie in this year' , 'type': 'positive' },
    {'movie_review': 'cool my boyfriend also said the movie is cool' , 'type': 'positive' },
    {'movie_review': 'awesome of the awesome movie ever' , 'type': 'positive' },
    {'movie_review': 'shame I wasted money and time' , 'type': 'negative'},
    {'movie_review': 'regret on this move. I will never never what movie from this director' , 'type': 'negative'},
    {'movie_review': 'I do not like this movie' , 'type': 'negative'},
    {'movie_review': 'I do not like actors in this movie' , 'type': 'negative'},
    {'movie_review': 'boring boring sleeping movie' , 'type': 'negative'},
]
df = pd.DataFrame(review_list)

df['label'] = df['type'].map({"positive":1,"negative":0})
df_x = df["movie_review"]
df_y = df["label"]

cv = CountVectorizer()
x_traincv = cv.fit_transform(df_x)
encoded_input = x_traincv.toarray()

# cv.get_feature_names()
# cv.inverse_transofrm(encoded_input[0])

mnb = MultinomialNB()
y_train = df_y.astype('int')
mnb.fit(x_traincv,y_train)

test_feedback_list = [
    {'movie_review': 'great great great movie ever', 'type': 'positive'},
    {'movie_review': 'I like this amazing movie', 'type': 'positive'},
    {'movie_review': 'my boyfriend said great movie ever', 'type': 'positive'},
    {'movie_review': 'cool cool cool', 'type': 'positive'},
    {'movie_review': 'awesome boyfriend said cool movie ever', 'type': 'positive'},
    {'movie_review': 'shame shame shame', 'type': 'negative'},
    {'movie_review': 'awesome director shame movie boring movie', 'type': 'negative'},
    {'movie_review': 'do not like this movie', 'type': 'negative'},
    {'movie_review': 'I do not like this boring movie', 'type': 'negative'},
    {'movie_review': 'awful terrible boring movie', 'type': 'negative'},
]

test_df = pd.DataFrame(test_feedback_list)
test_df['label'] = test_df['type'].map({"positive":1,"negative":0})

test_x = test_df["movie_review"]
test_y = test_df["label"]

x_testcv = cv.transform(test_x)
predictions = mnb.predict(x_testcv)

print(accuracy_score(test_y,predictions))




