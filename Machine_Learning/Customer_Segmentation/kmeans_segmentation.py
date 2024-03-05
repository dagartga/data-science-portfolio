import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# load the data
transactions = pd.read_excel('../../ABC_Grocery/grocery_database.xlsx', sheet_name='transactions')
product_areas = pd.read_excel('../../ABC_Grocery/grocery_database.xlsx', sheet_name='product_areas')

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
