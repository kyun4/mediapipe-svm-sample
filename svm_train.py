import pandas as pd
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the feature data from CSV
data = pd.read_csv('coco_features.csv')

# Split features and labels
X = data.drop(columns=['label'])  # Feature columns
y = data['label']  # Label column

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train SVM classifier
clf = svm.SVC(kernel='linear')
clf.fit(X_train, y_train)

# Test the model
y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

print(X)