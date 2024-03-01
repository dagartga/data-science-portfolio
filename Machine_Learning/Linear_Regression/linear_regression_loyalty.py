import pandas as pd
import pickle
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.utils import shuffle  
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_selection import RFECV


# load the data for modeling
with open('./data/df_with_loyalty.pkl', 'rb') as f:
    df_with_loyalty = pickle.load(f)
    
    
# drop the customer_id column
df_with_loyalty = df_with_loyalty.drop('customer_id', axis=1)

# shuffle the data
df_with_loyalty = shuffle(df_with_loyalty, random_state=42)

# drop missing values since only a few are present
df_with_loyalty = df_with_loyalty.dropna()


# create function to remove outliers based on 1.5 IQR
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
    df_with_loyalty = remove_outliers(df_with_loyalty, col)

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
model = LinearRegression()

# use the feature selection of recurvise feature elimination with cross validation
rfecv = RFECV(estimator=model)
fit = rfecv.fit(X_train, y_train)

optimal_feature_count = rfecv.n_features_
print(f'Total number of features: {X_train.shape[1]}')
print(f'Optimal number of features: {optimal_feature_count}')
print(f'Optimal features: {X_train.columns[fit.support_]}')

# create X_train and X_test with the optimal features
X_train = X_train.iloc[:, fit.support_]
X_test = X_test.iloc[:, fit.support_]

# train the model
model = LinearRegression()
model.fit(X_train, y_train)

# predict the test set
y_pred = model.predict(X_test)

# use r2 score to evaluate the model
r2 = r2_score(y_test, y_pred)
print(f'R2 score: {r2}')

# use adjusted r2 score to evaluate the model
n = X_test.shape[0]
p = X_test.shape[1]

def adjusted_r2_score(r2, n, p):
    return 1 - (1 - r2) * (n - 1) / (n - p - 1)

print(f'Adjusted R2 score: {adjusted_r2_score(r2, n, p)}')

# view the coefficients
coefs = pd.DataFrame({'variable': X_train.columns, 'coef': model.coef_})
coefs = coefs.sort_values(by='coef', ascending=False)

# plot the coefficients
plt.barh(coefs['variable'], coefs['coef'])
plt.xlabel('Coefficient')
plt.ylabel('Variable')
plt.title('Coefficients of the Linear Regression Model')
plt.show()

# save the model
with open('./models/linear_regression_loyalty.pkl', 'wb') as f:
    pickle.dump(model, f)