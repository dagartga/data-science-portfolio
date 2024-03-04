import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.feature_selection import RFECV


# load the data for modeling
with open('./data/abc_classification_modeling.pkl', 'rb') as f:
    df_signup = pickle.load(f)
    
    
# drop the customer_id column
df_signup = df_signup.drop('customer_id', axis=1)

# shuffle the data
df_signup = shuffle(df_signup, random_state=42)

# drop missing values since only a few are present
df_signup = df_signup.dropna()


# create function to remove outliers based on 2.0 IQR
def remove_outliers(df, column):
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 2.0 * iqr
    upper_bound = q3 + 2.0 * iqr
    df = df[(df[column] > lower_bound) & (df[column] < upper_bound)]
    return df

# drop oultiers
for col in ['distance_from_store', 'total_sales', 'total_items']:
    df_signup = remove_outliers(df_signup, col)

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

# min max scale the data between 0 and 1
scaler = MinMaxScaler()
X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)


# use the feature selection of recurvise feature elimination with cross validation
rf_clf = RandomForestClassifier(random_state=42)
rfecv = RFECV(estimator=rf_clf)
fit = rfecv.fit(X_train, y_train)

optimal_feature_count = rfecv.n_features_
print(f'Total number of features: {X_train.shape[1]}')
print(f'Optimal number of features: {optimal_feature_count}')
print(f'Optimal features: {X_train.columns[fit.support_]}')


# get the mean cross validation score
grid_scores = [np.mean(vals) for vals in fit.grid_scores_]

# plot the feature selection
plt.plot(range(1, len(fit.grid_scores_) + 1), grid_scores, marker='o')
plt.ylabel('R2 Score')
plt.xlabel('Number of Features')
plt.title(f'Feature Selection using RFE \n Optimal number of features is {optimal_feature_count} (at score of {round(rfecv.grid_scores_.max(), 2)})')
plt.tight_layout()
plt.show()

# create X_train and X_test with the optimal features
X_train = X_train.iloc[:, fit.support_]
X_test = X_test.iloc[:, fit.support_]

# train the model
model = KNeighborsClassifier()
model.fit(X_train, y_train)

# predict the test set
y_pred = model.predict(X_test)

# predict the probabilities
y_pred_prob = model.predict_proba(X_test)[:, 1]



# create confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
# plot the confusion matrix
plt.style.use('seaborn-poster')
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

# get the best probability threshold
thresholds = np.arange(0, 1, 0.01)

precision_scores = []
recall_scores = []
f1_scores = []

for thresh in thresholds:
    y_pred_thresh = (y_pred_prob > thresh).astype(int)
    precision = precision_score(y_test, y_pred_thresh, zero_division=0)
    recall = recall_score(y_test, y_pred_thresh)
    f1 = f1_score(y_test, y_pred_thresh)
    precision_scores.append(precision)
    recall_scores.append(recall)
    f1_scores.append(f1)

# get the max f1 score
max_f1 = max(f1_scores)
max_f1_index = f1_scores.index(max_f1)
best_thresh = thresholds[max_f1_index]
print(f'Best Threshold: {best_thresh}')

# plot the tresholds
plt.style.use('seaborn-poster')
plt.plot(thresholds, precision_scores, label='Precision', linestyle='--')
plt.plot(thresholds, recall_scores, label='Recall', linestyle='--')
plt.plot(thresholds, f1_scores, label='F1', linewidth=5)
plt.title(f'Finding the Optimal Threshold for Classification Model \n Max F1: {round(max_f1, 2)} (Threshold at {round(best_thresh, 2)})')
plt.xlabel('Threshold')
plt.ylabel('Assessment Score')
plt.legend(loc='lower left')
plt.tight_layout()
plt.show()

# generate predictions using the best threshold
y_pred_thresh = (y_pred_prob >= best_thresh).astype(int)

