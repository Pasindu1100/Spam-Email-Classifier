messages = [
    "Free entry in 2 a weekly competition!",
    "Hey, are we still meeting today?",
    "Congratulations, you won a prize!",
    "Can you call me back later?"
]

labels = [1, 0, 1, 0]  # 1 = spam, 0 = ham (not spam)

# These are give on Actual Dataset. we just need to Labels that only

#Step 2 : Clean the Text
#  usually lowercase everything and remove punctuation:

import re

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)  # remove punctuation/numbers
    return text

messages = [clean_text(msg) for msg in messages]
print(messages)

#Step 3: Convert Words → Numbers
# Use Bag of Words with CountVectorizer:

from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(messages)

print(vectorizer.get_feature_names_out())
print(X.toarray())



#Step 4: Train a Simple Model
#Now train Logistic Regression:

from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=200)
model.fit(X, labels)

# Predict on a new message
new_msg = ["Win a free prize now!"]
new_msg_clean = [clean_text(new_msg[0])]
new_X = vectorizer.transform(new_msg_clean)

print("Prediction:", model.predict(new_X))


