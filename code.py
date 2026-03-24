#Import all necessary Libararies
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer #Pre Defined Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Load the breast cancer dataset
data = load_breast_cancer()

X = data.data
y = data.target
#Print Shape
print("Shape of X:", X.shape)
print("Shape of y:", y.shape) #Shape has 1 column, which is expected for a target variable

#Create a DataFrame for better visualization
df = pd.DataFrame(X, columns=data.feature_names)
df['target'] = y

#Data Preprocessing/EDA
print("Dataset Head:")
print(df.head())

print("\nDataset Info:")
print(df.info())

print("\nSummary Statistics:")
print(df.describe()) 

print("\nMissing Values:")
print(df.isnull().sum().sort_values(ascending=False).head(20))

#Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,y,test_size=0.2, random_state=42,
    stratify=y #to maintain the same distribution of classes in train and test sets
    )

#Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#Train the Logistic Regression model
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

#Predictions
y_pred = model.predict(X_test_scaled)
y_pred_prob = model.predict_proba(X_test_scaled)[:, 1] #probability of the positive class (malignant)

#Check first 5 predicted probabilities
print("\nPredicted Probabilities for the first 5 test samples:")
print(y_pred_prob[:5])

'''Interpretation (simple language):
The predicted probabilities for the first 5 test samples indicate how likely each sample is to 
be classified as malignant (1) or benign (0). 
Malignant means the cancer is more likely to spread to other parts of the body, and it is generally more aggressive than benign tumors.
A probability close to 1 suggests a higher likelihood of being malignant, 
while a probability close to 0 suggests a higher likelihood of being benign. 
For example, if the predicted probability for a sample is 0.95, it means there is a 95% chance 
that the sample is malignant. 
Conversely, if the predicted probability is 0.05, it means there is only a 5% chance that the sample 
is malignant, and it is more likely to be benign. In this case, the predicted probabilities for the 
first 5 test samples are all very close to either 0 or 1, indicating strong confidence in the 
model's predictions for those samples.
'''

#Evaluation
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nAccuracy Score:")
print(accuracy_score(y_test, y_pred))

#Visualization
#Confusion Matrix Heatmap
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix Heatmap')
# plt.savefig("Confusion Matrix Heatmap.jpg") #Uncheck if you want to save the graph
plt.show()


'''Conclusion:
The model achieved an accuracy of approximately 97.37%, which indicates that it is performing
very well in classifying the breast cancer samples as malignant or benign.
The confusion matrix shows that there are 71 true positives (malignant samples correctly classified),
42 true negatives (benign samples correctly classified), 2 false positives (benign samples incorrectly classified as malignant), and 0 false negatives (malignant samples incorrectly classified as benign).
The classification report provides additional insights into the model's performance, showing high precision, recall, and F1-score for both classes.
Overall, the logistic regression model appears to be effective in distinguishing between malignant and benign breast cancer samples.
'''
