import streamlit as st
import pandas as pd
import pydeck as pdk
import geopandas as gpd
import plotly.express as px
from model import OutageModel

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

# You might also have a state shapefile or derive state names:
# The county shapefile often includes “STATEFP” (state FIPS code).
# To map state FIPS to state names, you can load a lookup table.
state_fips_to_name = {
    "01": "Alabama",
    "02": "Alaska",
    "04": "Arizona",
    # ... include all state codes ...
    "24": "Maryland",
    # etc.
}

# Clean categorical columns in your outage df
df["Event Type"] = df["Event Type"].astype(str).str.strip()
df["county"] = df["county"].astype(str).str.strip()
df["state"] = df["state"].astype(str).str.strip()

# Event type filter
event_types = sorted(df["Event Type"].unique())
selected_event = st.selectbox("Select Event Type", event_types)

# Filter the dataset
event_df = df[df["Event Type"] == selected_event].copy()

# Train and predict
model = OutageModel()
X_test, y_test = model.train(event_df)
event_df["Predicted_Customers"] = model.predict(event_df)

# Aggregate per county (keeping fips for merge)
county_df = event_df.groupby(["state", "county", "fips"], as_index=False).agg({
    "Predicted_Customers": "mean",
    "duration": "max"
})

# Define instability by duration
unstable_duration_threshold = 15  # (seconds, or adjust based on correct unit)
county_df["Unstable"] = county_df["duration"] > unstable_duration_threshold

# Merge with county shapefile to get names and centroids
county_df = county_df.merge(
    counties[["GEOID", "NAME", "STATEFP", "Latitude", "Longitude"]],
    left_on="fips",
    right_on="GEOID",
    how="left"
)

# Rename columns for clarity
county_df = county_df.rename(columns={
    "NAME": "County_Name",
    "STATEFP": "State_FIPS"
})

# Map state FIPS to state names
county_df["State_Name"] = county_df["State_FIPS"].map(state_fips_to_name)

# Color function
def outage_color(row):
    return [255, 0, 0, 160] if row["Unstable"] else [0, 255, 0, 160]

county_df["color"] = county_df.apply(outage_color, axis=1)

# Build PyDeck map
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

# State-level ranking with Plotly
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

# Display county-level table
st.write(f"Predicted Outages for {selected_event} (County Level)")
st.dataframe(county_df[["State_Name", "County_Name", "Predicted_Customers", "duration", "Unstable"]])
