import pandas as pd
import pickle
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.utils import shuffle  
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import r2_score
from sklearn.preprocessing import OneHotEncoder


# load the data for modeling
with open('./data/df_with_loyalty.pkl', 'rb') as f:
    df_with_loyalty = pickle.load(f)
    
    
# drop the customer_id column
df_with_loyalty = df_with_loyalty.drop('customer_id', axis=1)

# shuffle the data
df_with_loyalty = shuffle(df_with_loyalty, random_state=42)

# drop missing values since only a few are present
df_with_loyalty = df_with_loyalty.dropna()


# create X and y
X = df_with_loyalty.drop('customer_loyalty_score', axis=1)
y = df_with_loyalty['customer_loyalty_score']

# split the data from trainig and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# one hot encode the categorical columns
cat_cols = ['gender']
one_hot = OneHotEncoder(drop='first', sparse_output=False)
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

# create the Linear Regression model
model = DecisionTreeRegressor(random_state=42)

# train the model
model.fit(X_train, y_train)

# predict the test set
y_pred = model.predict(X_test)

# use r2 score to evaluate the model
r2 = r2_score(y_test, y_pred)
print(f'R2 score: {r2}')

# cross validate the model
cv = KFold(n_splits=4, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='r2')
print('Cross Validation Scores:', cv_scores.mean())

# use adjusted r2 score to evaluate the model
n = X_test.shape[0]
p = X_test.shape[1]

# create the adjusted r2 score function
def adjusted_r2_score(r2, n, p):
    return 1 - (1 - r2) * (n - 1) / (n - p - 1)

print(f'Adjusted R2 score: {adjusted_r2_score(r2, n, p)}')

# find best max depth
max_depths = range(1, 9)
r2_scores = []
for depth in max_depths:
    model = DecisionTreeRegressor(max_depth=depth, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    r2_scores.append(r2)
    
# find the best max depth
best_r2_score = max(r2_scores) 
best_max_depth = max_depths[r2_scores.index(best_r2_score)]   
    
# plot the max depth vs r2 score
plt.plot(max_depths, r2_scores)
plt.scatter(best_max_depth, best_r2_score, color='red', marker='x', label='Best Max Depth')
plt.xlabel('Max Depth')
plt.ylabel('R2 Score')
plt.title(f'Max Depth vs R2 Score \nBest Max Depth: {best_max_depth} \nBest R2 Score: {round(best_r2_score, 3)}')
plt.tight_layout()
plt.show()

## while the max depth of 8 gives the best r-squared, the score is farily flat starting at max depth of 4
# use max depth of 4
model = DecisionTreeRegressor(max_depth=4, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
print(f'R2 score: {r2}') # performance barely changes with less chance of overfitting

# plot the model
plt.figure(figsize=(20, 10))
plot_tree(model, filled=True, rounded=True, feature_names=X_train.columns, fontsize=16)


# save the model
with open('./models/decision_tree_regression_loyalty.pkl', 'wb') as f:
    pickle.dump(model, f)