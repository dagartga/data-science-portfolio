import pandas as pd
import pickle

# load in the data to be predicted
with open('./data/df_without_loyalty.pkl', 'rb') as f:
    df_to_predict = pickle.load(f)
    
# import the model and one hot encoder
with open('./models/random_forest_regression_loyalty.pkl', 'rb') as f:
    model = pickle.load(f)
    
with open('./models/one_hot_encoder_loyalty.pkl', 'rb') as f:
    one_hot = pickle.load(f)
    
# drop the customer_id column
df_to_predict = df_to_predict.drop('customer_id', axis=1)

# drop any missing values rows
df_to_predict = df_to_predict.dropna()

# one hot encode the categorical columns
cat_cols = ['gender']
transformed_array = one_hot.transform(df_to_predict[cat_cols])
encoded_feat_names = one_hot.get_feature_names_out(cat_cols)
trans_df = pd.DataFrame(transformed_array, columns=encoded_feat_names)
df_to_predict = pd.concat([df_to_predict.reset_index(drop=True), trans_df.reset_index(drop=True)], axis=1)
df_to_predict.drop(cat_cols, axis=1, inplace=True)

# make predictions
predictions = model.predict(df_to_predict)

# add the predictions to the dataframe
df_to_predict['customer_loyalty_score'] = predictions

# pickle the dataframe
with open('./data/df_with_loyalty_predictions.pkl', 'wb') as f:
    pickle.dump(df_to_predict, f)

