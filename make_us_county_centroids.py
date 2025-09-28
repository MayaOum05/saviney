import geopandas as gpd
import pandas as pd

counties = gpd.read_file(
    "https://www2.census.gov/geo/tiger/GENZ2022/shp/cb_2022_us_county_5m.zip"
)

counties["GEOID"] = counties["GEOID"].astype(str).str.zfill(5)

counties["Latitude"] = counties.geometry.centroid.y
counties["Longitude"] = counties.geometry.centroid.x

centroids = counties[["GEOID", "NAME", "STATEFP", "Latitude", "Longitude"]]

centroids.to_csv("data/us_county_centroids.csv", index=False)

print("us_county_centroids.csv saved to data/")
