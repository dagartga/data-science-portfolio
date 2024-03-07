import pandas as pd
from apyori import apriori

# load the data
df = pd.read_csv('../data/sample_data_apriori.csv')

# drop transaction_id column
df = df.drop('transaction_id', axis=1)

# create a list of lists for apriori
transactions_list = []

for i, row in df.iterrows():
    transaction = list(row.dropna())
    transactions_list.append(transaction)
    
# create the apriori model
association_rules = apriori(transactions_list, 
                            min_support=0.003, 
                            min_confidence=0.2, 
                            min_lift=3, 
                            min_length=2,
                            max_length=2)

# convert the association rules to a list
apriori_rules = list(association_rules)

# get a list of the rules
product1 = [list(rule[2][0][0])[0] for rule in apriori_rules]
product2 = [list(rule[2][0][1])[0] for rule in apriori_rules]
support = [rule[1] for rule in apriori_rules]
confidence = [rule[2][0][2] for rule in apriori_rules]
lift = [rule[2][0][3] for rule in apriori_rules]

# create a dataframe of the rules
apriori_rules_df = pd.DataFrame({'product1': product1, 
                         'product2': product2, 
                         'support': support, 
                         'confidence': confidence, 
                         'lift': lift})

# sort rules by lift
apriori_rules_df = apriori_rules_df.sort_values(by='lift', ascending=False)

# view New Zealand wines association rules
apriori_rules_df[apriori_rules_df['product1'].str.contains('New Zealand')]
