#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib


# In[4]:


# Load Dataset
url = './kidney_disease.csv'
df = pd.read_csv(url)

# Preview
print("Initial data snapshot:")
print(df.head())


# In[5]:


# Clean column names
df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')


# In[7]:


# Replace missing values ('?') with NaN
df = df.replace('?', np.nan)

# Convert all columns to appropriate types
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors='ignore')


# In[8]:


# Fill missing values
df.fillna(method='ffill', inplace=True)


# In[9]:


# Encode categorical features
label_encoders = {}
for col in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le


# In[10]:


# Features and target
X = df.drop('classification', axis=1)
y = df['classification']


# In[11]:


# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# In[18]:


# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=0)


# In[19]:


# Random Forest Model
model = RandomForestClassifier(n_estimators=150, max_depth=10, random_state=42)
model.fit(X_train, y_train)


# In[20]:


# Predictions
y_pred = model.predict(X_test)

# Evaluation
print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# In[21]:


print("Confusion Matrix:")
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")


# In[22]:


# Save model and scaler
# joblib.dump(model, 'random_forest_kidney_model.pkl')
# joblib.dump(scaler, 'scaler.pkl')


# In[23]:


from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, X_scaled, y, cv=5)
print("CV Accuracy:", scores)
print("Mean CV Accuracy:", scores.mean())


# In[24]:


import pickle

# Save the model in .sav format
filename = 'random_forest_kidney_model.sav'
pickle.dump(model, open(filename, 'wb'))

# Save the scaler too, if needed
scaler_filename = 'scaler.sav'
pickle.dump(scaler, open(scaler_filename, 'wb'))


# In[ ]:




