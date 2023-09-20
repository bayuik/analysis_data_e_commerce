# %% [markdown]
# # Proyek Analisis Data: Nama dataset
# - Nama: Bayu Indra Kusuma
# - Email: bayuindrakusuma05@gmail.com
# - Id Dicoding: https://www.dicoding.com/users/bayuik

# %% [markdown]
# ## Menentukan Pertanyaan Bisnis

# %% [markdown]
# - Bagaimana pola pembelian berdasarkan kategori produk dalam suatu wilayah geografis?
# - Bagaimana hubungan antara berat produk dan biaya pengiriman mempengaruhi nilai kepuasan pelanggan ?

# %% [markdown]
# ## Menyiapkan semua library yang dibutuhkan

# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# %% [markdown]
# ## Data Wrangling

# %% [markdown]
# ### Gathering Data

# %% [markdown]
# #### Customers Dataset

# %%
customers_dataset = pd.read_csv('datasets/customers_dataset.csv')
customers_dataset.head()

# %% [markdown]
# #### Orders Dataset

# %%
orders_dataset = pd.read_csv('datasets/orders_dataset.csv')
orders_dataset.head()

# %% [markdown]
# #### Order Items Dataset

# %%
order_items_dataset = pd.read_csv('datasets/order_items_dataset.csv')
order_items_dataset.head()

# %% [markdown]
# #### Order Reviews Dataset

# %%
order_reviews_dataset = pd.read_csv('datasets/order_reviews_dataset.csv')
order_reviews_dataset.head()

# %% [markdown]
# #### Products Dataset

# %%
products_dataset = pd.read_csv('datasets/products_dataset.csv')
products_dataset.head()

# %% [markdown]
# ### Assessing Data

# %%
customers_dataset.info()
print(f"duplicated: {customers_dataset.duplicated().sum()}")

# %%
orders_dataset.info()
print(f"duplicated: {orders_dataset.duplicated().sum()}")

# %%
order_items_dataset.info()
print(f"duplicated: {order_items_dataset.duplicated().sum()}")

# %%
order_reviews_dataset.info()
print(f"duplicated: {order_reviews_dataset.duplicated().sum()}")

# %%
products_dataset.info()
print(f"duplicated: {products_dataset.duplicated().sum()}")

# %% [markdown]
# ### Cleaning Data

# %%
customer_df = customers_dataset[['customer_id', 'customer_city']]
customer_df.head()

# %%
orders_df = orders_dataset[['order_id', 'customer_id', 'order_purchase_timestamp']]
orders_df.head()

# %%
order_items_df = order_items_dataset[['order_id', 'product_id', 'freight_value', 'price']]
order_items_df.head()

# %%
order_reviews_df = order_reviews_dataset[['review_id', 'order_id', 'review_score']]
order_reviews_df.head()

# %%
products_df = products_dataset[['product_id', 'product_category_name', 'product_weight_g']]
products_df.head()

# %% [markdown]
# #### Merge all datasets

# %%
df = pd.merge(customer_df, orders_df, on='customer_id')
df = pd.merge(df, order_items_df, on='order_id')
df = pd.merge(df, order_reviews_df, on='order_id')
df = pd.merge(df, products_df, on='product_id')
df.head()

# %% [markdown]
# #### Save all data to csv

# %%
df.to_csv('datasets/ecommerce_dataset.csv', index=False)

# %% [markdown]
# ## Exploratory Data Analysis (EDA)

# %% [markdown]
# #### Explore Data

# %%
df.describe(include='all')

# %% [markdown]
# #### Explore product weight and review score

# %%
df.groupby(by='review_score').agg({
    'customer_id': 'nunique',
    'product_weight_g': ["max", "min", "mean", "std"]
}).reset_index()

# %% [markdown]
# #### Explore freight value and review score

# %%
df.groupby(by='review_score').agg({
    'customer_id': 'nunique',
    'freight_value': ["max", "min", "mean", "std"]
}).reset_index()

# %% [markdown]
# #### Explore order rank in each state

# %%
df.groupby(by='customer_city').order_id.nunique().sort_values(ascending=False)

# %% [markdown]
# 

# %% [markdown]
# #### Explore product category and review score

# %%
df.groupby(by=['product_category_name']).agg({
    'product_id': 'nunique',
    'review_score': 'mean'
}).reset_index()

# %% [markdown]
# ## Visualization & Explanatory Analysis

# %% [markdown]
# ### Pertanyaan 1:

# %%
by_city = df.groupby(['customer_city', 'product_category_name']).size().reset_index(name='purchase_count')
top_category_per_city = by_city.sort_values(by=['customer_city', 'purchase_count'], ascending=[
                                            True, False]).groupby('customer_city').first().reset_index()
plt.figure(figsize=(10, 5))
sns.barplot(
    x='customer_city',
    y='purchase_count',
    hue='product_category_name',
    data=top_category_per_city.nlargest(10, 'purchase_count'))
plt.title('Most Purchases by city')
plt.xticks(rotation=90)
plt.show()

# %% [markdown]
# ### Pertanyaan 2:

# %%
correlation_matrix = df[['product_weight_g', 'freight_value', 'review_score']].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation between Product Weight, Freight Value, and Customer Satisfaction Score')
plt.show()

# %% [markdown]
# ### RFM Analysis

# %%
rfm_df = df.groupby(by='customer_id', as_index=False).agg({
    "order_purchase_timestamp": "max",
    "order_id": "nunique",
    "price": "sum"
})
rfm_df.columns = ['customer_id', 'max_order_timestamp', 'frequency', 'monetary']
rfm_df['max_order_timestamp'] = pd.to_datetime(rfm_df['max_order_timestamp']) 
recent_date = pd.to_datetime(df['order_purchase_timestamp']).max()
rfm_df['recency'] = rfm_df['max_order_timestamp'].apply(lambda x: (recent_date - x).days)

rfm_df.drop("max_order_timestamp", axis=1, inplace=True)
rfm_df.head()

# %%
fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(30, 6))

colors = ["#72BCD4", "#72BCD4", "#72BCD4", "#72BCD4", "#72BCD4"]

sns.barplot(y="recency", x="customer_id", data=rfm_df.sort_values(
    by="recency", ascending=True).head(5), palette=colors, ax=ax[0])
ax[0].set_ylabel(None)
ax[0].set_xlabel(None)
ax[0].set_title("By Recency (days)", loc="center", fontsize=18)
ax[0].tick_params(axis='x', labelsize=15)

sns.barplot(y="frequency", x="customer_id", data=rfm_df.sort_values(
    by="frequency", ascending=False).head(5), palette=colors, ax=ax[1])
ax[1].set_ylabel(None)
ax[1].set_xlabel(None)
ax[1].set_title("By Frequency", loc="center", fontsize=18)
ax[1].tick_params(axis='x', labelsize=15)

sns.barplot(y="monetary", x="customer_id", data=rfm_df.sort_values(
    by="monetary", ascending=False).head(5), palette=colors, ax=ax[2])
ax[2].set_ylabel(None)
ax[2].set_xlabel(None)
ax[2].set_title("By Monetary", loc="center", fontsize=18)
ax[2].tick_params(axis='x', labelsize=15)

plt.suptitle("Best Customer Based on RFM Parameters (customer_id)", fontsize=20)
plt.show()

# %% [markdown]
# ## Conclusion

# %% [markdown]
# ### Conclution pertanyaan 1

# %% [markdown]
# Berdasarkan hasil analisis data, dapat disimpulkan bahwa pola pembelian berdasarkan kategori produk dalam suatu wilayah geografis menunjukkan bahwa produk dengan kategori "cama_mesa_banho," "beleza_saude," dan "moveis_decoracao" memiliki jumlah pembelian yang signifikan di berbagai wilayah geografis. Pembelian terbanyak terjadi di kota Sao Paulo, diikuti oleh Rio De Janeiro dan Belo Horizonte.

# %% [markdown]
# ### conclution pertanyaan 2

# %% [markdown]
# Berdasarkan hasil analisis data, dapat disimpulkan bahwa berat produk tidak memiliki pengaruh yang signifikan terhadap review kepuasan pelanggan, dengan korelasi sebesar -0.03. Demikian pula, harga pengiriman juga tidak memiliki pengaruh yang besar terhadap review kepuasan konsumen, dengan korelasi sebesar -0.04.


