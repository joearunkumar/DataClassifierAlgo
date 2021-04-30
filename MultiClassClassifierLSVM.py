import joblib
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

# -----------------------------------------------------------------#
# Import data
data = pd.read_csv('Filtered_data.csv')

# Select required columns
data = data[['Comments', 'Classification']]

categories = ['generic','services','self-service','customer service','deals','product info','shipping','policy']

# -----------------------------------------------------------------#
# Linear Support Vector Machine classifier

# X -> features, y -> label
X = data.Comments
y = data.Classification

# dividing X, y into train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=data[['Classification']])

# training a Linear Support Vector Machine classifier
sgd = Pipeline([('vect', CountVectorizer()),
                ('tfidf', TfidfTransformer()),
                ('clf', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, random_state=42, max_iter=10, tol=None)),
                ])
sgd.fit(X_train, y_train)
print("Model trained")
joblib.dump(sgd, "Linear_Support_Vector_Classification_Model.sav")
print("Model saved")

# -----------------------------------------------------------------#
# Evaluate Model
y_pred = sgd.predict(X_test)
print('accuracy %s' % accuracy_score(y_pred, y_test))
print(classification_report(y_test, y_pred, target_names=categories))
