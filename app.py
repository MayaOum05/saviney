import streamlit as st
import pandas as pd
import pydeck as pdk
import numpy as np

st.title("Energy Grid Stability Prediction Map")

GRID_WIDTH = 10
GRID_HEIGHT = 10

np.random.seed(42)
data = []
for y in range(GRID_HEIGHT):
    for x in range(GRID_WIDTH):
        status = np.random.choice(["green", "yellow", "red", "white"], p=[0.1, 0.1, 0.1, 0.7])
        data.append({
            "x": x,
            "y": y,
            "status": status
        })

df = pd.DataFrame(data)

color_map = {
    "white": [255, 255, 255],
    "green": [0, 255, 0],
    "yellow": [255, 255, 0],
    "red": [255, 0, 0]
}
df["color"] = df["status"].map(color_map)

layer = pdk.Layer(
    "ScatterplotLayer",
    data=df,
    get_position=["x", "y"],
    get_color="color",
    get_radius=0.4,
    pickable=True
)

view_state = pdk.ViewState(
    longitude=GRID_WIDTH / 2,
    latitude=GRID_HEIGHT / 2,
    zoom=4,
    pitch=0,
    bearing=0
)


r = pdk.Deck(
    layers=[layer],
    initial_view_state=view_state,
    tooltip={"text": "{status}"}
)

st.pydeck_chart(r)
