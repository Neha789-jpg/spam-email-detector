
import pandas as pd

data = pd.read_csv(
    "sms+spam+collection/SMSSpamCollection",
    sep='\t',
    names=['label', 'message']
)

print("Dataset loaded successfully!")
print(data.head())
print(data.shape)

print(data.info())
print(data['label'].value_counts())

#Checking missing values

print(data.isnull().sum())

#Converting labels to numbers

data['label']=data['label'].map({'ham':0, 'spam':1})
print(data.head())

#Separating features and target

X = data['message']
Y = data['label']

#Applying TF-IDF Vectorizer

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(X)
print(X.shape)
print("TF-IDF Vectorization successful!")

#Splitting the dataset into training and testing sets

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2, random_state = 42)

print(X_train.shape)
print(X_test.shape)

from sklearn.linear_model import LogisticRegression

model = LogisticRegression()

model.fit(X_train, Y_train)

print("Model trained successfully!")

y_pred = model.predict(X_test)

print(y_pred[:10])


from sklearn.metrics import accuracy_score

accuracy = accuracy_score(Y_test, y_pred)

print("Accuracy:", accuracy)

import pickle

# Save trained model
with open('spam_model.pkl', 'wb') as file:
    pickle.dump(model, file)

# Save TF-IDF vectorizer
with open('vectorizer.pkl', 'wb') as file:
    pickle.dump(vectorizer, file)

print("Model and vectorizer saved successfully!")

from sklearn.metrics import classification_report

print(classification_report(Y_test, y_pred))