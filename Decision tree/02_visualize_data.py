import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import data


train_df = pd.DataFrame(data.district_dict_list)
train_df = train_df[['district','longitude','latitude','label']]

print(train_df.head())

# fit_reg = False 일시 회귀직선없이 산점도만 얻을수있음
# scatter_kws -> 점하고 사이즈 말함
# hue ->
sns.lmplot('longitude','latitude',data=train_df,fit_reg=False,
scatter_kws={'s':150},
markers=['o','x','+','*'],
hue="label")

# title
plt.title('district visualization in 2d plane')

plt.show()