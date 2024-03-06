import pandas as pd # pandas<2.0.0,>=1.0.0
import numpy as np
import pickle
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.utils import shuffle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.inspection import permutation_importance



# load the data for modeling
with open('./data/abc_classification_modeling.pkl', 'rb') as f:
    df_signup = pickle.load(f)
    

    
    
# drop the customer_id column
df_signup = df_signup.drop('customer_id', axis=1)

# shuffle the data
df_signup = shuffle(df_signup, random_state=42)

# drop missing values since only a few are present
df_signup = df_signup.dropna(how='any')



# create X and y
X = df_signup.drop('signup_flag', axis=1)
y = df_signup['signup_flag']

# split the data from trainig and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# one hot encode the categorical columns
cat_cols = ['gender']
one_hot = OneHotEncoder(drop='first', sparse=False)
one_hot.fit(X_train[cat_cols])
X_train_encoded = one_hot.transform(X_train[cat_cols])
encoded_feat_names = one_hot.get_feature_names_out(cat_cols)
X_train_encoded = pd.DataFrame(X_train_encoded, columns=encoded_feat_names)
X_train = pd.concat([X_train.reset_index(drop=True), X_train_encoded.reset_index(drop=True)], axis=1)
X_train.drop(cat_cols, axis=1, inplace=True)

# encode the test data
X_test_encoded = one_hot.transform(X_test[cat_cols])
X_test_encoded = pd.DataFrame(X_test_encoded, columns=encoded_feat_names)
X_test = pd.concat([X_test.reset_index(drop=True), X_test_encoded.reset_index(drop=True)], axis=1)
X_test.drop(cat_cols, axis=1, inplace=True)


# train the model
model = RandomForestClassifier(n_estimators=500, random_state=42, max_features=5)
model.fit(X_train, y_train)

# predict the test set
y_pred = model.predict(X_test)

# predict the probabilities
y_pred_prob = model.predict_proba(X_test)[:, 1]



# create confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
# plot the confusion matrix
# plt.style.use('seaborn-poster')
plt.matshow(conf_matrix, cmap='coolwarm', alpha=0.7)
plt.gca().xaxis.tick_bottom()
plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
for (i, j), val in np.ndenumerate(conf_matrix):
    plt.text(j, i, f'{val}', ha='center', va='center', fontsize=20)
plt.show()

# accuracy score
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')

# precision score
precision = precision_score(y_test, y_pred)
print(f'Precision: {precision}')

# recall score
recall = recall_score(y_test, y_pred)
print(f'Recall: {recall}')

# f1 score
f1 = f1_score(y_test, y_pred)
print(f'F1: {f1}')

# view feature importance
feature_importance = pd.DataFrame(model.feature_importances_)
feature_names = pd.DataFrame(X.columns)
feature_importance_summary = pd.concat([feature_names, feature_importance], axis=1)
feature_importance_summary.columns = ['input_variable', 'feature_importance']
feature_importance_summary.sort_values(by='feature_importance', inplace=True)
# plot the feature importance
plt.barh(feature_importance_summary['input_variable'], feature_importance_summary['feature_importance'])
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Feature Importance - Random Forest Regression')
plt.tight_layout()
plt.show()

# view the permutation importance
perm_importance = permutation_importance(model, X_test, y_test, random_state=42, n_repeats=10)
permutation_importance = pd.DataFrame(perm_importance['importances_mean'])
features = pd.DataFrame(X.columns)
permutation_importance_summary = pd.concat([features, permutation_importance], axis=1)
permutation_importance_summary.columns = ['input_variable', 'permutation_importance']
permutation_importance_summary.sort_values(by='permutation_importance', inplace=True)

plt.barh(permutation_importance_summary['input_variable'], permutation_importance_summary['permutation_importance'])
plt.ylabel('Feature')
plt.title('Permutation Importance - Random Forest Regression')
plt.tight_layout()
plt.show()

# save the model
with open('./models/random_forest_regression_signup.pkl', 'wb') as f:
    pickle.dump(model, f)
    
# save the one hot encoder
with open('./models/one_hot_encoder_rf_signup.pkl', 'wb') as f:
    pickle.dump(one_hot, f)