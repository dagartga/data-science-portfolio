import pandas as pd # pandas<2.0.0,>=1.0.0
import numpy as np
import pickle
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.utils import shuffle
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.preprocessing import OneHotEncoder



# load the data for modeling
with open('./data/abc_classification_modeling.pkl', 'rb') as f:
    df_signup = pickle.load(f)
    

    
    
# drop the customer_id column
df_signup = df_signup.drop('customer_id', axis=1)

# shuffle the data
df_signup = shuffle(df_signup, random_state=42)

# drop missing values since only a few are present
df_signup = df_signup.dropna()



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
model = DecisionTreeClassifier(random_state=42, max_depth=5)
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

# find the best max_depth
max_depths = list(range(1, 15))
f1_results = []
for max_depth in max_depths:
    model = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    f1 = f1_score(y_test, y_pred)
    f1_results.append(f1)
    
max_f1 = max(f1_results)
best_max_depth = max_depths[f1_results.index(max_f1)]
print(f'Best max_depth: {best_max_depth}')
print(f'Best F1: {max_f1}')

# plot the f1 results for max_depth
plt.plot(max_depths, f1_results)
plt.scatter(best_max_depth, max_f1, color='red', marker='x')
plt.title(f'F1 Score vs Max Depth\n Optimal Max Depth: {best_max_depth} (F1: {round(max_f1,2)})')
plt.xlabel('Max Depth of Decision Tree')
plt.ylabel('F1 Score')
plt.tight_layout()
plt.show()

# plot the decision tree
plt.figure(figsize=(25, 15))
tree = plot_tree(model, 
                 feature_names=X_train.columns, 
                 filled=True,
                 rounded=True,
                 fontsize=16)