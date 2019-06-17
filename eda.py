import pandas as pd

# Load training data frame
train_data_path = '..\\dat\\train.tsv'
train_df = pd.read_table(train_data_path)
print('Training DataFrame loaded')

cat_clusters = train_df.groupby('category_name')
print(cat_clusters)
print(len(cat_clusters))
print(cat_clusters.count())

# Filter prices per each category name
import matplotlib.pyplot as plt
for name, group in cat_clusters:
    print('Cluster: ', str(name))

    # Stats about every group price
    print(group.price.describe())

    # cluster_prices = group.price.tolist()
    # print(cluster_prices)
    # plt.hist(cluster_prices, normed=True, bins=30)



    # %matplotlib inline
    group.price.hist(bins=10, range=[0, 300])