import streamlit as st
import pandas as pd
import pydeck as pdk
import geopandas as gpd
import plotly.express as px
from model import OutageModel
from genai_utils import explain_grid_instability

st.set_page_config(page_title="Energy Grid Stability Predictions", layout="wide")
st.title("Energy Grid Stability Predictions")

df = pd.read_csv("data/2023-outages.csv")

counties = gpd.read_file(
    "https://www2.census.gov/geo/tiger/GENZ2022/shp/cb_2022_us_county_5m.zip"
)

df["fips"] = df["fips"].astype(str).str.zfill(5)
counties["GEOID"] = counties["GEOID"].astype(str).str.zfill(5)
counties["Latitude"] = counties.geometry.centroid.y
counties["Longitude"] = counties.geometry.centroid.x

state_fips_to_name = {
    "01": "Alabama","02": "Alaska","04": "Arizona","05": "Arkansas","06": "California",
    "08": "Colorado","09": "Connecticut","10": "Delaware","12": "Florida","13": "Georgia",
    "15": "Hawaii","16": "Idaho","17": "Illinois","18": "Indiana","19": "Iowa","20": "Kansas",
    "21": "Kentucky","22": "Louisiana","23": "Maine","24": "Maryland","25": "Massachusetts",
    "26": "Michigan","27": "Minnesota","28": "Mississippi","29": "Missouri","30": "Montana",
    "31": "Nebraska","32": "Nevada","33": "New Hampshire","34": "New Jersey","35": "New Mexico",
    "36": "New York","37": "North Carolina","38": "North Dakota","39": "Ohio","40": "Oklahoma",
    "41": "Oregon","42": "Pennsylvania","44": "Rhode Island","45": "South Carolina","46": "South Dakota",
    "47": "Tennessee","48": "Texas","49": "Utah","50": "Vermont","51": "Virginia","53": "Washington",
    "54": "West Virginia","55": "Wisconsin","56": "Wyoming"
}

df["Event Type"] = df["Event Type"].astype(str).str.strip()
df["county"] = df["county"].astype(str).str.strip()
df["state"] = df["state"].astype(str).str.strip()

event_mapping = {
    "Severe Weather": "Severe Weather",
    "Natural Disaster": "Severe Weather",
    "Weather": "Severe Weather",
    "Physical Attack": "Physical Attack",
    "Cyber Event": "Cyber Event",
    "Transmission Interruption": "Transmission Interruption"
}
df["Event Simplified"] = df["Event Type"].map(event_mapping).fillna("Other")

filters, main = st.columns([1, 3])

with filters:
    st.header("Filters")
    event_types = sorted(df["Event Simplified"].unique())
    selected_event = st.selectbox("Select Event Type", event_types)

    all_counties = df["county"].unique()
    search_county = st.text_input("Search County", "")

event_df = df[df["Event Simplified"] == selected_event].copy()

model = OutageModel()
X_test, y_test = model.train(event_df)
event_df["Predicted_Customers"] = model.predict(event_df)

county_df = event_df.groupby(["state", "county", "fips"], as_index=False).agg({
    "Predicted_Customers": "mean",
    "duration": "max"
})

county_df = county_df.merge(
    counties[["GEOID", "NAME", "STATEFP", "Latitude", "Longitude"]],
    left_on="fips",
    right_on="GEOID",
    how="left"
)

county_df = county_df.rename(columns={"NAME": "County_Name","STATEFP": "State_FIPS"})
county_df["State_Name"] = county_df["State_FIPS"].map(state_fips_to_name)

def classify_grid(duration):
    if duration <= 15:
        return "Stable"
    elif duration <= 60:
        return "Unstable"
    else:
        return "Critical"

county_df["Grid_Status"] = county_df["duration"].apply(classify_grid)

status_colors = {
    "Stable": [0, 200, 0, 180],
    "Unstable": [255, 200, 0, 180],
    "Critical": [200, 0, 0, 180]
}
county_df["color"] = county_df["Grid_Status"].map(status_colors)

layer = pdk.Layer(
    "ColumnLayer",
    data=county_df.dropna(subset=["Latitude", "Longitude"]),
    get_position=["Longitude", "Latitude"],
    get_elevation="Predicted_Customers",
    elevation_scale=30,
    radius=20000,
    get_fill_color="color",
    pickable=True,
    auto_highlight=True,
    extruded=True,
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
    map_style="mapbox://styles/mapbox/light-v9",
    tooltip={
        "text": "State: {State_Name}\nCounty: {County_Name}\nStatus: {Grid_Status}\nPredicted Customers: {Predicted_Customers}\nDuration: {duration} min"
    }
)

with main:
    st.subheader(f"Map of {selected_event} Grid Impacts")
    st.pydeck_chart(r)

    st.subheader(f"State Rankings and Generated Explainations for {selected_event}")
    col1, col2 = st.columns([2, 1])

    with col1:
        state_df = county_df.groupby(["State_Name", "Grid_Status"], as_index=False)["Predicted_Customers"].sum()
        state_df = state_df.sort_values("Predicted_Customers", ascending=False).dropna()
        fig = px.bar(
            state_df,
            x="Predicted_Customers",
            y="State_Name",
            color="Grid_Status",
            orientation="h",
            title=f"Predicted Customers by State and Status for {selected_event}",
            labels={"Predicted_Customers": "Predicted Customers", "State_Name": "State"},
            barmode="stack"
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        filtered_county_df = county_df.copy()
        if search_county:
            filtered_county_df = county_df[county_df["County_Name"].str.contains(search_county, case=False, na=False)]

        for _, row in filtered_county_df.iterrows():
            st.markdown(f"**{row['County_Name']}, {row['State_Name']}**")
            st.write(f"Predicted Customers: {row['Predicted_Customers']}")
            st.write(f"Duration: {row['duration']} minutes")
            st.write(f"Grid Status: {row['Grid_Status']}")
            if row["Grid_Status"] != "Stable":
                explanation = explain_grid_instability(
                    county_name=row["County_Name"],
                    state_name=row["State_Name"],
                    predicted_customers=row["Predicted_Customers"],
                    duration=row["duration"],
                    event_type=selected_event
                )
                st.info(explanation)
            st.markdown("---")
