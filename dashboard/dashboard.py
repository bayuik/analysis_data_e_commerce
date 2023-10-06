import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('dashboard/main_data.csv')
rfm_df = pd.read_csv('dashboard/rfm.csv')


def create_by_city(df):
    top_category = df.groupby(['product_category_name'])[
        'order_id'].count().reset_index(name='purchase_count')
    top_category = top_category.sort_values(
        by='purchase_count', ascending=False)
    return top_category


def create_correlation(df, selected_variables):
    selected_variables = [x.lower().replace(' ', '_') for x in selected_variables]
    correlation_maxtrix = df[selected_variables].corr()
    return correlation_maxtrix


with st.sidebar:
    city = st.selectbox('Select City', df['customer_city'].unique())
    corr_variables = st.multiselect(
        'Select Correlation Variables', options=('Product Weight G', 'Freight Value', 'Review Score'))

main_df = df[df['customer_city'] == city]

st.header('E-Commerce Dashboard')
st.subheader('Top Order by City')
fig, ax = plt.subplots(figsize=(20, 15))
sns.barplot(x='product_category_name', y='purchase_count',
            data=create_by_city(main_df).nlargest(10, 'purchase_count'))
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
    

st.subheader('RFM Analysis')
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
st.pyplot(plt)