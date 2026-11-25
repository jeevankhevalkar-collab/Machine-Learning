import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.metrics import (
    root_mean_squared_error,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

import pickle

st.set_page_config(
    page_title="My Wide App",
    layout="wide",  # Options: "centered" (default) or "wide"
    initial_sidebar_state="expanded",
)

st.title("Diamond_price Prediction")

# Data Load
df = pd.read_csv(r"D:\tops\python env\ML\data\diamonds.csv")
df = df.drop(columns=["Unnamed: 0"])

num_col_eda = ["carat", "depth", "table", "price", "x", "y", "z"]
cat_col_eda = ["cut", "color", "clarity"]


# st.write(df.head())
# st.table(df.head())
# st.dataframe(df.head())

b1 = st.sidebar.checkbox("Show Dataset")
if b1:
    st.write(df.head())

# EDA:
col1, col2 = st.columns([1, 1])
b2 = st.sidebar.checkbox("EDA")
if b2:
    s1 = st.sidebar.selectbox(
        "Types of EDA", ["Univariate", "Bivariate", "Multivariate"]
    )
    if s1 == "Univariate":
        col_name = st.sidebar.selectbox("Select columnn", df.columns)
        if col_name in num_col_eda:
            chart1 = px.histogram(df[col_name])
            col1.plotly_chart(chart1)

            chart2 = px.box(df[col_name])
            col2.plotly_chart(chart2)

        elif col_name in cat_col_eda:
            df_cat_eda = df.groupby(col_name)["price"].count().reset_index()
            # df_cat_eda.rename(columns={"price":"count"},inplace=True)
            # st.write(df_cat_eda)
            chart1 = px.bar(df_cat_eda, col_name, "price")
            col1.plotly_chart(chart1)

            chart2 = px.pie(df_cat_eda, col_name, "price")
            col2.plotly_chart(chart2)

    elif s1 == "Bivariate":
        col_name = st.sidebar.selectbox("Select Column", df.columns)
        target = df["price"]

        if col_name in num_col_eda:
            chart1 = px.scatter(df, col_name, "price")
            col1.plotly_chart(chart1)

            num_col_eda = df.groupby(col_name)["price"].mean().reset_index()

            chart2 = px.line(num_col_eda, col_name, "price")
            col2.plotly_chart(chart2)

    else:
        pass
    st.write(s1)

x = df.drop(columns="price")
y = df["price"]

test_size = st.sidebar.slider("Select Test Size", 0.10, 0.90, 0.20, 0.05)
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=test_size, random_state=5
)

b3 = st.sidebar.checkbox("check shape")
if b3:
    st.write(f"x train {x_train.shape}")
    st.write(f"x test {x_test.shape}")
    st.write(f"y train {y_train.shape}")
    st.write(f"y test {y_test.shape}")

cat_col = ["cut", "color", "clarity"]
num_col = ["carat", "depth", "table", "x", "y", "z"]

num_train = x_train[num_col]
num_test = x_test[num_col]

cat_train = x_train[cat_col]
cat_test = x_test[cat_col]

scaler = StandardScaler()
num_train_s = scaler.fit_transform(num_train)
num_train_s = pd.DataFrame(num_train_s, columns=num_col, index=num_train.index)

num_test_s = scaler.transform(num_test)
num_test_s = pd.DataFrame(num_test_s, columns=num_col, index=num_test.index)

encoder = OrdinalEncoder(
    categories=[
        ["Fair", "Good", "Very Good", "Premium", "Ideal"],
        ["J", "I", "H", "G", "F", "E", "D"],
        ["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"],
    ]
)
cat_train_e = encoder.fit_transform(cat_train)
cat_train_e = pd.DataFrame(cat_train_e, columns=cat_col, index=cat_train.index)


cat_test_e = encoder.transform(cat_test)
cat_test_e = pd.DataFrame(cat_test_e, columns=cat_col, index=cat_test.index)

x_train_new = num_train_s.join(cat_train_e)

x_test_new = num_test_s.join(cat_test_e)

# st.dataframe(x_train_new)

models = [
    "Linear Regression",
    "Ridge",
    "Lasso",
    "Decision Tree Regressor",
    "Random Forest Regressor",
    "Gradient Boosting Regressor",
    "Ada Boost Regressor",
    "SVR",
    "KNeighbors Regressor",
    "Polynomial Features",
]


model_dict = {
    "Linear Regression": LinearRegression(),
    "Ridge": Ridge(),
    "Lasso": Lasso(),
    "Decision Tree Regressor": DecisionTreeRegressor(),
    "Random Forest Regressor": RandomForestRegressor(),
    "Gradient Boosting Regressor": GradientBoostingRegressor(),
    "Ada Boost Regressor": AdaBoostRegressor(),
    "SVR": SVR(),
    "KNeighbors Regressor": KNeighborsRegressor(),
}

model_name = st.sidebar.selectbox("Select Model", list(model_dict.keys()))
selected_model = model_dict[model_name]

# Training Block
if st.sidebar.button("Train Model"):
    selected_model.fit(x_train_new, y_train)

    y_pred = selected_model.predict(x_test_new)

    rmse = root_mean_squared_error(y_pred, y_test)
    rae = mean_absolute_error(y_pred, y_test)
    msqe = mean_squared_error(y_pred, y_test)
    r2s = r2_score(y_pred, y_test)

    metrics_data = {
        "Metric": [
            "Root Mean Squared Error (RMSE)",
            "Mean Absolute Error (MAE)",
            "Mean Squared Error (MSE)",
            "R² Score",
        ],
        "Value": [f"{rmse:.2f}", f"{rae:.2f}", f"{msqe:.2f}", f"{r2s:.2f}"],
    }

    metrics_df = pd.DataFrame(metrics_data)
    st.table(metrics_df)
    # Calculate performance scores using correct variables
    train_score = selected_model.score(x_train_new, y_train)
    test_score = selected_model.score(x_test_new, y_test)

    models = {"Pred_model": models, "Scaler": scaler, "Encoder": encoder}
    # Convert dictionary into pickle bytes
    model_bytes = pickle.dumps(models)

    # Download button
    st.download_button(
        label="Download Models", data=model_bytes, file_name="models.pkl"
    )
