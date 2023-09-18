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
# #### Customer Dataset

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
customers_dataset.drop(
    ['customer_unique_id', 'customer_zip_code_prefix', 'customer_state'], axis=1, inplace=True)
customers_dataset.head()

# %%
orders_dataset.drop(['order_status', 'order_purchase_timestamp', 'order_approved_at', 'order_delivered_carrier_date',
                    'order_delivered_customer_date', 'order_estimated_delivery_date'], axis=1, inplace=True)
orders_dataset

# %%
order_items_dataset.drop(
    ['order_item_id', 'seller_id', 'shipping_limit_date', 'price'], axis=1, inplace=True)
order_items_dataset.head()

# %%
order_reviews_dataset.drop(['review_comment_title', 'review_creation_date',
                           'review_answer_timestamp', 'review_comment_message'], axis=1, inplace=True)
order_reviews_dataset.head()

# %%
products_dataset.drop(['product_name_lenght', 'product_description_lenght', 'product_photos_qty',
                      'product_length_cm', 'product_height_cm', 'product_width_cm'], axis=1, inplace=True)
products_dataset.head()

# %% [markdown]
# ## Exploratory Data Analysis (EDA)

# %%
df = pd.merge(customers_dataset, orders_dataset, on='customer_id')
df = pd.merge(df, order_items_dataset, on='order_id')
df = pd.merge(df, order_reviews_dataset, on='order_id')
df = pd.merge(df, products_dataset, on='product_id')
df

# %%
df.describe(include='all')

# %%
df.groupby(by='review_score').agg({
    'customer_id': 'nunique',
    'product_weight_g': ["max", "min", "mean", "std"]
})

# %%
df.groupby(by='review_score').agg({
    'customer_id': 'nunique',
    'freight_value': ["max", "min", "mean", "std"]
}).reset_index()

# %%
df.groupby(by='customer_city').order_id.nunique().sort_values(ascending=False)

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
# ## Conclusion

# %% [markdown]
# ### Conclution pertanyaan 1

# %% [markdown]
# Berdasarkan hasil analisis data, dapat disimpulkan bahwa pola pembelian berdasarkan kategori produk dalam suatu wilayah geografis menunjukkan bahwa produk dengan kategori "cama_mesa_banho," "beleza_saude," dan "moveis_decoracao" memiliki jumlah pembelian yang signifikan di berbagai wilayah geografis. Pembelian terbanyak terjadi di kota Sao Paulo, diikuti oleh Rio De Janeiro dan Belo Horizonte.

# %% [markdown]
# ### conclution pertanyaan 2

# %% [markdown]
# Berdasarkan hasil analisis data, dapat disimpulkan bahwa berat produk tidak memiliki pengaruh yang signifikan terhadap review kepuasan pelanggan, dengan korelasi sebesar -0.03. Demikian pula, harga pengiriman juga tidak memiliki pengaruh yang besar terhadap review kepuasan konsumen, dengan korelasi sebesar -0.04.


