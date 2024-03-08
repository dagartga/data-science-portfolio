import pandas as pd
from causalimpact import CausalImpact # pip install pycausalimpact

# import the data
transactions = pd.read_excel('../data/ABC_Grocery_Data/grocery_database.xlsx', sheet_name='transactions')
campaign_data = pd.read_excel('../data/ABC_Grocery_Data/grocery_database.xlsx', sheet_name='campaign_data')

# aggregate the sales data by day
customer_daily_sales = transactions.groupby(['customer_id', 'transaction_date'])['sales_cost'].sum().reset_index()

# merge with the campaign data
customer_daily_sales = customer_daily_sales.merge(campaign_data, on='customer_id', how='inner')

# create pivot table
causal_impact_df = customer_daily_sales.pivot_table(index='transaction_date', 
                                                    columns='signup_flag', 
                                                    values='sales_cost', 
                                                    aggfunc='mean')

# create daily frequency
causal_impact_df.index.freq = "D"

# put the positive group as the first column
causal_impact_df = causal_impact_df[[1, 0]]

# rename the columns
causal_impact_df.columns = ['member', 'non_member'] 

# create start and end period
pre_period = ['2020-04-01', '2020-06-30']
post_period = ['2020-07-01', '2020-09-30']

# create the causal impact model
ci = CausalImpact(causal_impact_df, pre_period, post_period)

# plot the results
ci.plot()

# print the summary
print(ci.summary())

# print the report
print(ci.summary(output='report'))