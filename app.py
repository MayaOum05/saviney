import streamlit as st
import pandas as pd
import pydeck as pdk
from model import OutageModel
import numpy as np
from geopy.geocoders import Nominatim
import time
import os
import json

st.title("Power Outage Prediction and Visualization")

df = pd.read_csv("data/grid.csv")

model = OutageModel()
X_test, y_class_test, y_reg_test = model.train(df)

# Predictions
outage_probs, preds = model.predict(df)
df["Outage_Prob"] = outage_probs
df["Predicted_Customers"] = preds

# --- Geocoding ---
cache_file = "geo_cache.json"
if os.path.exists(cache_file):
    with open(cache_file, "r") as f:
        geo_cache = json.load(f)
else:
    geo_cache = {}

geolocator = Nominatim(user_agent="power_outage_app")

def geocode_area(area):
    if area in geo_cache:
        return geo_cache[area]
    try:
        location = geolocator.geocode(area)
        if location:
            coord = (location.latitude, location.longitude)
            geo_cache[area] = coord
            time.sleep(0.1)  # be gentle with API
            return coord
    except:
        pass
    # fallback to somewhere in the US if geocoding fails
    geo_cache[area] = (37.8, -95)
    return (37.8, -95)

df["Latitude"], df["Longitude"] = zip(*df["Geographic Areas"].apply(geocode_area))

# Save cache
with open(cache_file, "w") as f:
    json.dump(geo_cache, f)

# --- Scale for column height ---
df["Elevation"] = df["Predicted_Customers"] / df["Predicted_Customers"].max() * 50000

# --- Color function ---
def outage_color(prob):
    if prob < 0.33:
        return [0, 255, 0, 160]
    elif prob < 0.66:
        return [255, 255, 0, 160]
    else:
        return [255, 0, 0, 160]

df["Color"] = df["Outage_Prob"].apply(outage_color)

# --- PyDeck ColumnLayer ---
layer = pdk.Layer(
    "ColumnLayer",
    data=df,
    get_position=["Longitude", "Latitude"],
    get_elevation="Elevation",
    elevation_scale=1,
    radius=40000,
    get_fill_color="Color",
    pickable=True,
    auto_highlight=True,
)

view_state = pdk.ViewState(
    latitude=df["Latitude"].mean(),
    longitude=df["Longitude"].mean(),
    zoom=4,
    pitch=45,
)

r = pdk.Deck(
    layers=[layer],
    initial_view_state=view_state,
    tooltip={"text": "Area: {Geographic Areas}\nRegion: {NERC Region}\nOutage Probability: {Outage_Prob}\nPredicted Customers: {Predicted_Customers}"}
)

st.pydeck_chart(r)

st.write("Predictions vs Actuals:")
st.dataframe(df[["Geographic Areas", "Number of Customers Affected", "Predicted_Customers", "Outage_Prob"]])
