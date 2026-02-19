import pandas as pd

#Load Data Set

df = pd.read_csv("spam.csv", encoding="latin-1")

#Keep only the relevant Columns
df = df[['v1','v2']]

df.columns = ['label','message']

print(df.head())

# Step 2: Preprocess Labels
# Convert spam and ham into numeric values.

df['label'] = df['label'].map({'ham': 0, 'spam':1})


# Step 3: Text Preprocessing
# Turn text into numeric features using Bag of Words

from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(stop_words='english')  # remove common words
X = vectorizer.fit_transform(df['message'])
y = df['label']


# Step 4: Train-Test Split
# Split into training and testing sets.

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 5: Train Logistic Regression
# Train the model on the training data.

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))


# Step 6: Evaluate Further
# Check precision, recall, and F1-score for deeper insight.

from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred, target_names=['ham', 'spam']))
