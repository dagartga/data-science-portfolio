import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

# Load data
df = pd.read_csv('./data/pipeline_data.csv')

# Split data
X = df.drop('purchase', axis=1)
y = df['purchase']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# create list of numeric features
numeric_features = ['age', 'credit_score']
categorical_features = ['gender']

# Create pipeline for numeric features
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer()),
    ('scaler', StandardScaler())])

# create pipeline for categorical features
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='U')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

# preprocessing pipeline
preprocessor = ColumnTransformer(transformers=[('numeric', numeric_transformer, numeric_features),
                                               ('categorical', categorical_transformer, categorical_features)])

## Logistic Regression
# create classifier pipeline
clf = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier', LogisticRegression(random_state=42))])

# fit the model
clf.fit(X_train, y_train)

# create predictions
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')

## Random Forest
# create classifier pipeline
clf = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier', RandomForestClassifier(random_state=42))])

# fit the model
clf.fit(X_train, y_train)

# create predictions
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')

# save the pipeline
joblib.dump(clf, './data/rf_pipeline.joblib')

