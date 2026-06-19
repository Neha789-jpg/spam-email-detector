import pickle

# Load model
with open('spam_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Load vectorizer
with open('vectorizer.pkl', 'rb') as file:
    vectorizer = pickle.load(file)

message = ["Congratulations! You have won a free iPhone. Claim now!"]

message_tfidf = vectorizer.transform(message)

prediction = model.predict(message_tfidf)

if prediction[0] == 1:
    print("Spam")
else:
    print("Not Spam")