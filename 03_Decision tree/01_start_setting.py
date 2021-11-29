import pandas as pd
import data

train_df = pd.DataFrame(data.district_dict_list)
train_df = train_df[['district','longitude','latitude','label']]

test_df = pd.DataFrame(data.dong_dict_list)
test_df = test_df[['dong','longitude','latitude','label']]

print(train_df.label.value_counts())
print(test_df.label.value_counts())

#print(train_df.head())
print(train_df.columns)
print(train_df.index)