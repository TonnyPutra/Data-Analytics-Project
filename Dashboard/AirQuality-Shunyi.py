import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import matplotlib.patches as mpatches

# Helper function
def reg(df):
    # mean x & y
    x = df["PM2.5"]
    y = df["PM10"]
    n = np.size(x)
    m_x = np.mean(x)
    m_y = np.mean(y)

    # cross-deviation & deviation about y
    SS_xy = np.sum(y * x) - n * m_y * m_x
    SS_xx = np.sum(x * x) - n * m_x * m_x

    # calculating regression coefficients
    b_1 = SS_xy / SS_xx
    b_0 = m_y - b_1 * m_x

    # plotting the actual points as scatter plot
    plt.scatter(x, y, marker="o", s=0.2)

    # predicted response vector
    y_pred = b_0 + b_1 * x
    return y_pred, b_0, b_1


def cor(df):
    Cor = df.corr(numeric_only=True)
    return Cor["PM2.5"][6]


def pca(df):
    # pollutant as feature
    Pollutant = df.loc[:, "PM2.5":"CO"]
    # hour as target
    Hour = df["Daytime"]

    # standarizing feature
    Pollutant = (Pollutant - Pollutant.mean()) / Pollutant.std()

    # searching eigen
    PolCor = (1 / 150) * Pollutant.T.dot(Pollutant)
    u, s, v = np.linalg.svd(PolCor)
    eig_values, eig_vectors = s, u
    np.sum(eig_values)

    # principal component 1 and 2
    pc1 = Pollutant.dot(eig_vectors[:, 0])
    pc2 = Pollutant.dot(eig_vectors[:, 1])
    pca_df = pd.concat([pc1, pc2], axis=1)

    # mapping to target
    target_names = {0: "Morning", 1: "Afternoon", 2: "Evening", 3: "Night"}

    pca_df["target"] = Hour
    pca_df["target"] = pca_df["target"].map(target_names)
    return pc1, pc2, pca_df["target"]



# Load cleaned data
data = pd.read_csv('https://github.com/TonnyPutra/Data-Analytics-Project/blob/main/Dashboard/AirQualityClean.csv')
data["Date"] = pd.to_datetime(data["Date"])

# Filter data
min_date = data["Date"].min()
max_date = data["Date"].max()

with st.sidebar:
    # logo
    st.image("air.png")

    # determine start_date
    date = st.date_input(label="Date", min_value=min_date)

    main_df = data[(data["Date"] >= str(date)) & (data["Date"] <= str(date))]

    col1, col2 = st.columns(2)

    with col1:
        Avg_Temp = round(main_df["TEMP"].mean(), 3)
        st.metric("Average Temperature", value=Avg_Temp)

    with col2:
        Avg_Pres = round(main_df["PRES"].mean(), 3)
        st.metric("Average Air Pressure", value=Avg_Pres)

    col1, col2 = st.columns([1.2, 2])
    with col1:
        st.write(r"$\textsf{\Large Weather: }$")

    with col2:
        if main_df["RAIN"].sum() / len(main_df.index) == 0:
            st.image("sunny.png", width=75)
        elif (
            main_df["RAIN"].sum() / len(main_df.index) > 0
            and main_df["RAIN"].sum() / len(main_df.index) <= 1
        ):
            st.image("rain.png", width=75)
        elif (
            main_df["RAIN"].sum() / len(main_df.index) > 1
            and main_df["RAIN"].sum() / len(main_df.index) <= 5
        ):
            st.image("heavy_rain.png", width=75)
        else:
            st.image("storm.png", width=75)

st.header("Air Quality in Shunyi")
st.subheader("Ozone Levels Over Time")

fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(main_df["hour"], main_df["O3"], marker="o", linewidth=2, color="#90CAF9")
ax.set_ylabel("O3", size=20)
ax.set_xlabel("hour", size=20)
ax.tick_params(axis="y", labelsize=15)
ax.tick_params(axis="x", labelsize=15)

st.pyplot(fig)

st.subheader("PM2.5 and PM10 Relationship")

col1, col2 = st.columns([2, 1])
y_pred, a, b = reg(main_df)
with col1:
    fig, ax = plt.subplots(figsize=(12, 4))

    plt.scatter(main_df["PM2.5"], main_df["PM10"], marker="o", s=25)
    plt.plot(main_df["PM2.5"], y_pred, color="r")
    ax.set_title("PM2.5 vs PM10", loc="center", fontsize=50)
    ax.set_ylabel("PM10", size=35)
    ax.set_xlabel("PM2.5", size=35)
    ax.tick_params(axis="x", labelsize=25)
    ax.tick_params(axis="y", labelsize=25)

    st.pyplot(fig)

with col2:
    st.metric("Correlation value", value=round(cor(main_df), 3))
    st.write("Regression line: ")
    st.latex(
        r"""
y = {:.2f} + {:.2f} * x
""".format(
            a, b
        )
    )

st.subheader("Pollutant Cluster by Daytime")

pc1, pc2, target = pca(main_df)

colors = np.array(
    [
        (
            "#333A73"
            if x == "Night"
            else (
                "#FBA834"
                if x == "Morning"
                else "#387ADF" if x == "Afternoon" else "#50C4ED"
            )
        )
        for x in target
    ]
)
night_patch = mpatches.Patch(color="#333A73", label="Night")
morning_patch = mpatches.Patch(color="#FBA834", label="Morning")
afternoon_patch = mpatches.Patch(color="#387ADF", label="Afternoon")
evening_patch = mpatches.Patch(color="#50C4ED", label="Evening")
fig, ax = plt.subplots(figsize=(12, 4))
plt.scatter(pc1, pc2, c=colors)
ax.set_title("2D PCA Graph of Pollutant", loc="center", fontsize=30)
ax.set_ylabel("pc2", size=20)
ax.set_xlabel("pc1", size=20)
ax.tick_params(axis="x", labelsize=15)
ax.tick_params(axis="y", labelsize=15)
ax.legend(
    bbox_to_anchor=(1, 1),
    handles=[morning_patch, afternoon_patch, evening_patch, night_patch],
    loc=2,
)
st.pyplot(fig)

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    Avg_PM2_5 = round(main_df["PM2.5"].mean(), 3)
    st.metric("Average PM2.5", value=Avg_PM2_5)

with col2:
    Avg_PM10 = round(main_df["PM10"].mean(), 3)
    st.metric("Average PM10", value=Avg_PM10)

with col3:
    Avg_SO2 = round(main_df["SO2"].mean(), 3)
    st.metric("Average SO2", value=Avg_SO2)

with col4:
    Avg_NO2 = round(main_df["NO2"].mean(), 3)
    st.metric("Average NO2", value=Avg_NO2)

with col5:
    Avg_CO = round(main_df["CO"].mean(), 3)
    st.metric("Average CO", value=Avg_CO)
