import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('datasets/ecommerce_dataset.csv')


def create_by_city(df):
    top_category = df.groupby(['product_category_name'])[
        'order_id'].count().reset_index(name='purchase_count')
    top_category = top_category.sort_values(
        by='purchase_count', ascending=False)
    return top_category


def create_correlation(df, selected_variables):
    selected_variables = [x.lower().replace(' ', '_') for x in selected_variables]
    print(selected_variables)
    correlation_maxtrix = df[selected_variables].corr()
    return correlation_maxtrix


with st.sidebar:
    city = st.selectbox('Select City', df['customer_city'].unique())
    corr_variables = st.multiselect(
        'Select Correlation Variables', options=('Product Weight G', 'Freight Value', 'Review Score'))

main_df = df[df['customer_city'] == city]
by_city_df = create_by_city(main_df)

st.header('E-Commerce Dashboard')
st.subheader('Top Order by City')
fig, ax = plt.subplots(figsize=(20, 15))
sns.barplot(x='product_category_name', y='purchase_count',
            data=by_city_df.nlargest(10, 'purchase_count'))
plt.title(f"Most Purchases in {city.title()}")
plt.xticks(rotation=45)
plt.xlabel('Product Category')
plt.ylabel('Purchase Count')
st.pyplot(fig)

st.subheader('Correlation Matrix')
if len(corr_variables) >= 2:
    plt.figure(figsize=(10, 8))
    correlation_matrix = create_correlation(main_df, corr_variables)
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title(f"Correlation between {corr_variables}")
    st.pyplot(plt)
else:
    st.write('Please select at leaest 2 variables')