import streamlit as st
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

# Streamlit title and image
st.title('Cars')
st.image("cars.jpg", width=500)
st.title('Cars Dataset')

# Load the data
data = pd.read_csv("Car data.csv")
st.write("Shape of the dataset:", data.shape)

# Sidebar menu
menu = st.sidebar.radio("Menu", ["Home", "Vehicle Name"])

if menu == "Home":
    st.image("22222.jpg", width=550)
    st.header("Tabular Data of a Vehicle")
    if st.checkbox("Tabular Data"):
        st.table(data.head(150))
    
    st.header("Statistical Summary of the DataFrame")
    if st.checkbox("Statistics"):
        st.table(data.describe())
    
    st.header("Correlation Graph")
    if st.checkbox("Correlation Graph"):
        numeric_data = data.select_dtypes(include=[np.number]).copy()
        numeric_data = numeric_data.apply(pd.to_numeric, errors='coerce')
        numeric_data = numeric_data.dropna(axis=1, how='all')
        corr_matrix = numeric_data.corr()
        
        if corr_matrix.empty or corr_matrix.isnull().all().all():
            st.write("No valid numeric data available for correlation.")
        else:
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", ax=ax)
            st.pyplot(fig)
    
    st.title("Graphs")
    graph = st.selectbox("Different types of graphs", ["Scatter Plot", "Bar Graph", "Histogram"])
    
    if graph == "Scatter Plot":
        value = st.slider("Filter data using CC", 0, 300)
        data = data.loc[data["CC"] >= value]
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.scatterplot(data=data, x="Name", y="CC", hue="Type", ax=ax)
        st.pyplot(fig)

    if graph == "Bar Graph":
        fig, ax = plt.subplots(figsize=(3.5, 2))
        sns.barplot(x="Type", y=data.index, data=data, ax=ax)
        st.pyplot(fig)
    
    if graph == "Histogram":
        fig, ax = plt.subplots(figsize=(5, 3))
        sns.histplot(data["Type"], kde=True, ax=ax)
        st.pyplot(fig)

if menu == "Vehicle Name":
    st.title("Prediction of Car Prices")
    
    lr = LinearRegression()
    X = np.array(data["CC"]).reshape(-1, 1)
    y = np.array(data["Min Price (Lakh)"]).reshape(-1, 1)
    lr.fit(X, y)
    
    value = st.number_input("CC Value", min_value=int(X.min()), max_value=int(X.max()), step=1)
    value = np.array(value).reshape(1, -1)
    prediction = lr.predict(value)
    
    if st.button("Price Prediction"):
        st.write(f"Predicted Min Price (Lakh): {prediction[0][0]:.2f}")
