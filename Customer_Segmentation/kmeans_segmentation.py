import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# load the data
transactions = pd.read_excel('../data/ABC_Grocery_Data/grocery_database.xlsx', sheet_name='transactions')
product_areas = pd.read_excel('../data/ABC_Grocery_Data/grocery_database.xlsx', sheet_name='product_areas')

# concat the dataframes
df = pd.merge(transactions, product_areas, how='inner', on='product_area_id')

# drop any non-food categories
df = df[df['product_area_name'] != 'Non-Food']

# aggregate the data
df_summary = df.groupby(['customer_id', 'product_area_name'])['sales_cost'].sum().reset_index()

# use pivot table
df_pivot = df.pivot_table(index='customer_id', 
                                    columns='product_area_name', 
                                    values='sales_cost', 
                                    aggfunc='sum',
                                    fill_value=0,
                                    margins=True,
                                    margins_name='Total').rename_axis(None, axis=1)


# get percentage of sales for each product area
data_for_cluster = df_pivot.div(df_pivot['Total'], axis=0).drop('Total', axis=1)

# scale the data
scale_norm = MinMaxScaler()
data_for_cluster_scaled = scale_norm.fit_transform(data_for_cluster)
data_for_cluster_scaled = pd.DataFrame(data_for_cluster_scaled, columns=data_for_cluster.columns)

# use wcss to find the optimal number of clusters
wcss = []
k_list = list(range(1, 11))

for k in k_list:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(data_for_cluster_scaled)
    wcss.append(kmeans.inertia_)
    
# plot the elbow method
plt.plot(k_list, wcss)
plt.title('Within Cluster Sum of Squares - K Means')
plt.xlabel('K Clusters')
plt.ylabel('WCSS Score')
plt.tight_layout()
plt.show()

# instantiate the model and fit using k = 3
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(data_for_cluster_scaled)

# add the cluster labels to the original data
data_for_cluster['cluster'] = kmeans.labels_


# profile the clusters
cluster_profile = data_for_cluster.groupby('cluster')[['Dairy', 'Fruit', 'Meat', 'Vegetables']].mean().reset_index()   

# view the customer profile for each cluster and category
plt.bar(cluster_profile[cluster_profile['cluster'] == 0].columns[1:], cluster_profile[cluster_profile['cluster'] == 0].values[0][1:], color='blue', alpha=0.7)
plt.xlabel('Product Area')
plt.ylabel('Mean Sales %')
plt.title('Customer Purchasing Profile - Cluster 0')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# view the customer profile for each cluster and category
plt.bar(cluster_profile[cluster_profile['cluster'] == 1].columns[1:], cluster_profile[cluster_profile['cluster'] == 1].values[0][1:], color='blue', alpha=0.7)
plt.xlabel('Product Area')
plt.ylabel('Mean Sales %')
plt.title('Customer Purchasing Profile - Cluster 1')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# view the customer profile for each cluster and category
plt.bar(cluster_profile[cluster_profile['cluster'] == 2].columns[1:], cluster_profile[cluster_profile['cluster'] == 2].values[0][1:], color='blue', alpha=0.7)
plt.xlabel('Product Area')
plt.ylabel('Mean Sales %')
plt.title('Customer Purchasing Profile - Cluster 2')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# cluster 0 is the most balanced in terms of product area sales / No dietary restrictions
# cluster 1 looks to be vegan (no dairy no meat)
# cluster 2 looks to be vegetarian (no meat)