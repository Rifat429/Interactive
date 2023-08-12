import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split

lst=[] 
input_string = input()
lst.append(input_string)
with open('model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)


tfidf = TfidfVectorizer(analyzer = 'word', max_features = 150)
test = tfidf.fit_transform(lst)
test=test.toarray()

test = pd.DataFrame(test)
feature = tfidf.vocabulary_
col_names = []
for key, value in feature.items():
    print(key, ' : ', value)
    col_names.append(key)

test.columns = col_names
col_names
test
pred=model.predict(test)
print(pred)
