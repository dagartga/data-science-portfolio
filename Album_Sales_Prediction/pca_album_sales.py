import pandas as pd # pandas<2.0.0,>=1.0.0
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import PCA



# load the data for modeling
df_albums = pd.read_csv('../data/Album_Sales_Data/sample_data_pca.csv')

    
    
# drop the user_id column
df_albums = df_albums.drop('user_id', axis=1)

# shuffle the data
df_albums = shuffle(df_albums, random_state=42)

# drop missing values since only a few are present
df_albums = df_albums.dropna(how='any')



# create X and y
X = df_albums.drop('purchased_album', axis=1)
y = df_albums['purchased_album']

# split the data from trainig and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# apply PCA
pca = PCA(n_components=None, random_state=42)
pca.fit(X_train)

# view the explained variance ratio
explained_variance = pca.explained_variance_ratio_

# create the cumulative sum of the explained variance
cum_sum_explained_variance = explained_variance.cumsum()

# view the cumulative sum of the explained variance
num_vars_list = list(range(1, 101))

plt.figure(figsize=(15, 10))
plt.subplot(2, 1, 1)
plt.bar(num_vars_list, explained_variance)
plt.title('Variance across Principal Components')
plt.xlabel('Number of Components')
plt.ylabel('% Variance')
plt.tight_layout()

plt.subplot(2, 1, 2)
plt.plot(num_vars_list, cum_sum_explained_variance)
plt.title('Cumulative Variance across Principal Components')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative % Variance')
plt.tight_layout()
plt.show()

# apply PCA with 75% of the variance
pca = PCA(n_components=0.75, random_state=42)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

# train Random Forest Classifier
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

# run accuracy score because balanced classes
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.3f}')