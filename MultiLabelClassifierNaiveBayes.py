import joblib
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

# -----------------------------------------------------------------#
# Import data

# Import data from csv
data = pd.read_csv('Filtered_data.csv')

# Select required columns
data = data[['Comments', 'Classification']]

categories = ['generic','services','self-service','customer service','deals','product info','shipping','ispu','policy']

# -----------------------------------------------------------------#
# Naive Bayes classifier

# X -> features, y -> label
X = data.Comments
y = data.Classification
print(y.unique())

# dividing X, y into train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=data[['Classification']])

# training a Naive Bayes classifier
nb = Pipeline([('vect', CountVectorizer()),
               ('tfidf', TfidfTransformer()),
               ('clf', MultinomialNB()),
               ])

nb.fit(X_train, y_train)
print("Model trained")
joblib.dump(nb, "Naive_Bayes_Classification_Model.sav")
print("Model saved")

# -----------------------------------------------------------------#
# Evaluate Model

y_pred = nb.predict(X_test)
print('accuracy %s' % accuracy_score(y_pred, y_test))
print(classification_report(y_test, y_pred, target_names=categories))
