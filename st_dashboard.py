import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from babel.numbers import format_currency
sns.set(style='dark')

df = pd.read_csv('datasets/ecommerce_dataset.csv')


def create_by_city(df):
    top_category = df.groupby(['product_category_name'])['order_id'].count().reset_index(name='purchase_count')
    top_category = top_category.sort_values(by='purchase_count', ascending=False)
    return top_category

def create_correlation(df, corr1, corr2, corr3):
    correlation_maxtrix = df[[corr1, corr2, corr3]].corr()
    return correlation_maxtrix



with st.sidebar:
    city = st.selectbox('Select City', df['customer_city'].unique())
    corr1 = st.selectbox('Select Correlation 1', df.columns,index=df.columns.get_loc('product_weight_g'))
    corr2 = st.selectbox('Select Correlation 2', df.columns, index=df.columns.get_loc('freight_value'))
    corr3 = st.selectbox('Select Correlation 3', df.columns, index=df.columns.get_loc('review_score'))

main_df = df[df['customer_city'] == city]
correlation_matrix = create_correlation(df, corr1, corr2, corr3)
by_city_df = create_by_city(main_df)

st.header('E-Commerce Dashboard')
st.subheader('Top Order by City')
fig, ax = plt.subplots(figsize=(20, 15))
sns.barplot(x='product_category_name', y='purchase_count', data=by_city_df.nlargest(10, 'purchase_count'))
plt.title(f"Most Purchases in {city.title()}")
plt.xticks(rotation=45)
plt.xlabel('Product Category')
plt.ylabel('Purchase Count')
st.pyplot(fig)

st.subheader('Correlation Matrix')
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title(f"Correlation between {corr1}, {corr2}, and {corr3}")
plt.show()