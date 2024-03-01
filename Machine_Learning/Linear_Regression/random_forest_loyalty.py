import pandas as pd
import pickle
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
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
model = RandomForestRegressor(random_state=42)

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

# find best Estimator
estimators = range(10, 100, 10)
r2_scores = []
for est in estimators:
    model = RandomForestRegressor(n_estimators=est, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    r2_scores.append(r2)
    
# find the best estimator
best_r2_score = max(r2_scores) 
best_estimator = estimators[r2_scores.index(best_r2_score)]   
    
# plot the n_estimators vs r2 score
plt.plot(estimators, r2_scores)
plt.scatter(best_estimator, best_r2_score, color='red', marker='x', label='Best Estimator')
plt.xlabel('N Estimators')
plt.ylabel('R2 Score')
plt.title(f'Max Depth vs R2 Score \nBest Estimator: {best_estimator} \nBest R2 Score: {round(best_r2_score, 3)}')
plt.tight_layout()
plt.show()

## n_estimators of 20 is best
# create the Linear Regression model
model = RandomForestRegressor(n_estimators=20, random_state=42)

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

print(f'Adjusted R2 score: {adjusted_r2_score(r2, n, p)}')

# view feature importance
feature_importance = model.feature_importances_
feature_names = X_train.columns
feature_importance_df = pd.DataFrame({'feature': feature_names, 'importance': feature_importance})
feature_importance_df = feature_importance_df.sort_values('importance', ascending=True)
# plot the feature importance
plt.barh(feature_importance_df['feature'], feature_importance_df['importance'])
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Feature Importance - Random Forest Regression')
plt.tight_layout()
plt.show()

# save the model
with open('./models/random_forest_regression_loyalty.pkl', 'wb') as f:
    pickle.dump(model, f)