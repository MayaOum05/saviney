import streamlit as st
import pandas as pd
import pydeck as pdk
import geopandas as gpd
import plotly.express as px
from model import OutageModel
from genai_utils import explain_grid_instability

st.title("2023 Power Outage Predictions by Event Type")

# Load outage dataset
df = pd.read_csv("data/2023-outages.csv")

# Load US counties shapefile (with name attributes)
counties = gpd.read_file(
    "https://www2.census.gov/geo/tiger/GENZ2022/shp/cb_2022_us_county_5m.zip"
)

# Ensure FIPS codes properly zero-padded
df["fips"] = df["fips"].astype(str).str.zfill(5)
counties["GEOID"] = counties["GEOID"].astype(str).str.zfill(5)

# Extract centroids
counties["Latitude"] = counties.geometry.centroid.y
counties["Longitude"] = counties.geometry.centroid.x

# Map state FIPS to state names
state_fips_to_name = {
    "01": "Alabama",
    "02": "Alaska",
    "04": "Arizona",
    "05": "Arkansas",
    "06": "California",
    "08": "Colorado",
    "09": "Connecticut",
    "10": "Delaware",
    "12": "Florida",
    "13": "Georgia",
    "15": "Hawaii",
    "16": "Idaho",
    "17": "Illinois",
    "18": "Indiana",
    "19": "Iowa",
    "20": "Kansas",
    "21": "Kentucky",
    "22": "Louisiana",
    "23": "Maine",
    "24": "Maryland",
    "25": "Massachusetts",
    "26": "Michigan",
    "27": "Minnesota",
    "28": "Mississippi",
    "29": "Missouri",
    "30": "Montana",
    "31": "Nebraska",
    "32": "Nevada",
    "33": "New Hampshire",
    "34": "New Jersey",
    "35": "New Mexico",
    "36": "New York",
    "37": "North Carolina",
    "38": "North Dakota",
    "39": "Ohio",
    "40": "Oklahoma",
    "41": "Oregon",
    "42": "Pennsylvania",
    "44": "Rhode Island",
    "45": "South Carolina",
    "46": "South Dakota",
    "47": "Tennessee",
    "48": "Texas",
    "49": "Utah",
    "50": "Vermont",
    "51": "Virginia",
    "53": "Washington",
    "54": "West Virginia",
    "55": "Wisconsin",
    "56": "Wyoming"
}


# Clean categorical columns
df["Event Type"] = df["Event Type"].astype(str).str.strip()
df["county"] = df["county"].astype(str).str.strip()
df["state"] = df["state"].astype(str).str.strip()

# Event type filter
event_types = sorted(df["Event Type"].unique())
selected_event = st.selectbox("Select Event Type", event_types)

# Filter dataset
event_df = df[df["Event Type"] == selected_event].copy()

# Train and predict
model = OutageModel()
X_test, y_test = model.train(event_df)
event_df["Predicted_Customers"] = model.predict(event_df)

# Aggregate per county
county_df = event_df.groupby(["state", "county", "fips"], as_index=False).agg({
    "Predicted_Customers": "mean",
    "duration": "max"
})

# Define instability
unstable_duration_threshold = 15
county_df["Unstable"] = county_df["duration"] > unstable_duration_threshold

# Merge with county shapefile
county_df = county_df.merge(
    counties[["GEOID", "NAME", "STATEFP", "Latitude", "Longitude"]],
    left_on="fips",
    right_on="GEOID",
    how="left"
)

county_df = county_df.rename(columns={
    "NAME": "County_Name",
    "STATEFP": "State_FIPS"
})
county_df["State_Name"] = county_df["State_FIPS"].map(state_fips_to_name)

# Color function
county_df["color"] = county_df["Unstable"].apply(lambda x: [255, 0, 0, 160] if x else [0, 255, 0, 160])

# PyDeck map
layer = pdk.Layer(
    "ColumnLayer",
    data=county_df.dropna(subset=["Latitude", "Longitude"]),
    get_position=["Longitude", "Latitude"],
    get_elevation="Predicted_Customers",
    elevation_scale=0.5,
    radius=20000,
    get_fill_color="color",
    pickable=True,
    auto_highlight=True,
)

view_state = pdk.ViewState(
    latitude=county_df["Latitude"].mean(),
    longitude=county_df["Longitude"].mean(),
    zoom=4,
    pitch=45,
)

r = pdk.Deck(
    layers=[layer],
    initial_view_state=view_state,
    tooltip={
        "text": "State: {State_Name}\nCounty: {County_Name}\nPredicted: {Predicted_Customers}\nUnstable: {Unstable}"
    }
)

st.pydeck_chart(r)

# State-level ranking
state_df = county_df.groupby("State_Name", as_index=False)["Predicted_Customers"].sum()
state_df = state_df.sort_values("Predicted_Customers", ascending=False).dropna()

st.subheader(f"State Rankings for {selected_event}")
fig = px.bar(
    state_df,
    x="Predicted_Customers",
    y="State_Name",
    orientation="h",
    title=f"Predicted Customers by State for {selected_event}",
    labels={"Predicted_Customers": "Predicted Customers", "State_Name": "State"}
)
st.plotly_chart(fig, use_container_width=True)

# County-level table with AI explanations
st.subheader(f"Predicted Outages with Explanations for {selected_event}")
for _, row in county_df.iterrows():
    st.write(f"### {row['County_Name']}, {row['State_Name']}")
    st.write(f"**Predicted Customers:** {row['Predicted_Customers']}")
    st.write(f"**Duration:** {row['duration']} minutes")
    st.write(f"**Unstable:** {row['Unstable']}")
    if row["Unstable"]:
        explanation = explain_grid_instability(
            county_name=row["County_Name"],
            state_name=row["State_Name"],
            predicted_customers=row["Predicted_Customers"],
            duration=row["duration"],
            event_type=selected_event
        )
        st.write(f"**Explanation:** {explanation}")
