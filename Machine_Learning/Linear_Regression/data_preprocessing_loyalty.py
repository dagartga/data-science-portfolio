import pandas as pd
import pickle

# import the data
loyalty_scores = pd.read_excel('./data/grocery_database.xlsx', sheet_name='loyalty_scores')
customer_details = pd.read_excel('./data/grocery_database.xlsx', sheet_name='customer_details')
transactions = pd.read_excel('./data/grocery_database.xlsx', sheet_name='transactions')

# merge datasets
df = pd.merge(customer_details, loyalty_scores, on='customer_id', how='left')

# column customer_loyalty_score is the target

# aggregate the transactions data
sales_summary = transactions.groupby('customer_id').agg({'sales_cost': 'sum', 
                                                         'num_items': 'sum',
                                                         'transaction_id': 'count',
                                                         'product_area_id': 'nunique'}).reset_index()

# rename the aggregated columns
sales_summary.columns = ['customer_id', 'total_sales', 'total_items', 'transaction_count', 'product_area_count']

# create average basket amount
sales_summary['average_basket_value'] = sales_summary['total_sales'] / sales_summary['transaction_count']

df = pd.merge(df, sales_summary, on='customer_id', how='inner')

# split the dataframes into rows with loyalty scores and rows without
df_with_loyalty = df[df['customer_loyalty_score'].notnull()]
df_without_loyalty = df[df['customer_loyalty_score'].isnull()]
# drop the target column from the dataframe without loyalty scores
df_without_loyalty = df_without_loyalty.drop('customer_loyalty_score', axis=1)


# save the dataframes to pickle file
with open('./data/df_with_loyalty.pkl', 'wb') as f:
    pickle.dump(df_with_loyalty, f)
    
with open('./data/df_without_loyalty.pkl', 'wb') as f:
    pickle.dump(df_without_loyalty, f)
