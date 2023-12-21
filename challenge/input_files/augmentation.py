import pandas as pd

new = 'new_interactions.csv'
old = 'data_train.csv'

new_df = pd.read_csv(new)
old_df = pd.read_csv(old)

# sort the new_df by user_id
new_df = new_df.sort_values(by=['row'])

# merge the two dataframes removing duplicates
merged_df = pd.concat([old_df, new_df]).drop_duplicates(subset=['row', 'col'], keep=False)

# save the new dataframe
merged_df.to_csv('data_train_augmented.csv', index=False)

