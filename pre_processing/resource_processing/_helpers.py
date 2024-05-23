import atlite
import xarray as xr
import pandas as pd
import numpy as np
import geopandas as gpd
from geopy.distance import great_circle
from dask.distributed import Client, LocalCluster
import os


from shapely.ops import voronoi_diagram, nearest_points
from shapely.geometry import MultiPoint, Point
from scipy.interpolate import griddata
from IPython.display import clear_output


def load_supply_regions(filename):
    print("Reading supply regions from file: ", filename)
    regions={}
    for r in [1, 10, 27, 34, 159]:
        regions[r] = gpd.read_file(filename, layer = str(r))
        regions[r] = regions[r].to_crs("EPSG:4326")

    regions[10]["name"] = regions[10]["SupplyArea"]
    regions[34]["name"] = regions[34]["LocalArea"]
    regions[159]["name"] = regions[159]["MTS_1"]

    return regions

def load_gis_data(data_bundle_path):
    gis_data = {"supply_regions": {}}

    print("Loading Eskom Supply Regions from", f"{data_bundle_path}/rsa_supply_regions.gpkg")
    for r in [1, 10, 27, 34, 159]:
        gis_data["supply_regions"][r] = gpd.read_file(f"{data_bundle_path}/rsa_supply_regions.gpkg", layer = str(r))
        gis_data["supply_regions"][r] = gis_data["supply_regions"][r].to_crs("EPSG:4326")

    gis_data["supply_regions"][10]["name"] = gis_data["supply_regions"][10]["SupplyArea"]
    gis_data["supply_regions"][34]["name"] = gis_data["supply_regions"][34]["LocalArea"]
    gis_data["supply_regions"][159]["name"] = gis_data["supply_regions"][159]["MTS_1"]

    print("Loading EIA applications from", f"{data_bundle_path}/REEA_OR_2023_Q3.shp")
    eia_regions = gpd.read_file(f"{data_bundle_path}/REEA_OR_2023_Q3.shp").to_crs("EPSG:4326")
    eia_regions = eia_regions[eia_regions["TECHNOLOGY"].isin(["Wind", "Solar PV", "Wind and Solar PV"])]
    gis_data["eia_regions"] = eia_regions

    print("Loading REDZs from", f"{data_bundle_path}/REDZs.shp")
    redz1 = gpd.read_file(f"{data_bundle_path}/REDZs.shp").to_crs("EPSG:4326")
    redz2 = gpd.read_file(f"{data_bundle_path}/PHASE 2_REDZs.shp").to_crs("EPSG:4326")
    redz = pd.concat([redz1, redz2], ignore_index=True).to_crs("EPSG:4326")
    gis_data["redz"] = redz

    print("Loading Power Corridors from", f"{data_bundle_path}/Power_corridors.shp")
    corridors = gpd.read_file(f"{data_bundle_path}/Power_corridors.shp").to_crs("EPSG:4326")
    gis_data["corridors"] = corridors

    redz_corridors_eia = pd.concat([redz, corridors, eia_regions], ignore_index=True).to_crs("EPSG:4326")
    gis_data["redz_corridors_eia"] = redz_corridors_eia

    print("Loading SACAD from", f"{data_bundle_path}/SACAD_OR_2023_Q3.shp")
    sacad = gpd.read_file(f"{data_bundle_path}/SACAD_OR_2023_Q3.shp")
    sacad.to_crs("EPSG:4326", inplace=True)

    print("Loading SAPAD from", f"{data_bundle_path}/SAPAD_OR_2023_Q3.shp")
    sapad = gpd.read_file(f"{data_bundle_path}/SAPAD_OR_2023_Q3.shp")
    sapad.to_crs("EPSG:4326", inplace=True)

    print("Loading SKA exclusion from", f"{data_bundle_path}/SKA/SKA_exclusion.shp")
    ska = gpd.read_file(f"{data_bundle_path}/SKA/SKA_exclusion.shp")
    ska.to_crs("EPSG:4326", inplace=True)

    exclusion = pd.concat([sacad, sapad, ska], ignore_index=True) # combined sacad and sapad zones
    gis_data["exclusion"] = exclusion.to_crs("EPSG:4326")

    gis_data["data_bundle_path"] = data_bundle_path

    return gis_data



def set_availability(area, gis_data, cutout, supply_regions):

    data_bundle_path = gis_data["data_bundle_path"]
    supply_regions = supply_regions.set_index("name").rename_axis("bus")
    excluder = atlite.ExclusionContainer(crs=3035)
    salandcover_classes = pd.read_csv(f"{data_bundle_path}/salandcover_classes.csv", index_col=0)

    excluder.add_raster(
        f"{data_bundle_path}/SALandCover_OriginalUTM35North_2013_GTI_72Classes/sa_lcov_2013-14_gti_utm35n_vs22b.tif",
        codes=salandcover_classes[salandcover_classes["include_pv"]==1].index.to_list(),
        invert=True,
        crs=32635,
    )

    excluder.add_geometry(supply_regions.geometry, invert=True)
    if area != "all":
        excluder.add_geometry(gis_data[area].geometry, invert=True)

    excluder.add_geometry(gis_data["exclusion"].geometry)

    cluster = LocalCluster(n_workers=10, threads_per_worker=1)
    client = Client(cluster, asynchronous=True)

    return cutout.availabilitymatrix(supply_regions, excluder)


def reshape_xarray(da, cells):
    data = da.values
    data = data.reshape(len(da.time), len(cells.y.unique()), len(cells.x.unique()))

    return xr.DataArray(
                data = data,
                coords={"time": da.time, "lat": cells.y.unique(), "lon": cells.x.unique()},
                dims = ["time", "lat", "lon"],
            )

def generate_pv_timeseries(cutout, type, dc_ac_ratio, module):
    if type == "Fixed Tilt":
        timeseries = cutout.pv(
            panel="CSi",
            orientation="latitude_optimal",
            shapes=cutout.grid,
            tracking=None,
            per_unit=True,
        )
    elif type == "Single Axis":
        timeseries = cutout.pv(
            panel="CSi",
            orientation={"slope": 0, "azimuth": 0},
            shapes=cutout.grid,
            tracking="horizontal",
            per_unit=True,
        )
    elif type == "Rooftop":
        timeseries_east = cutout.pv(
            panel="CSi",
            orientation={"slope": 10, "azimuth": 15},
            shapes=cutout.grid,
            tracking=None,
            per_unit=True,
        )

        timeseries_west = cutout.pv(
            panel="CSi",
            orientation={"slope": 10, "azimuth": 345},
            shapes=cutout.grid,
            tracking=None,
            per_unit=True,
        )
        timeseries = (timeseries_east + timeseries_west)/2
    timeseries =  reshape_xarray((timeseries * dc_ac_ratio).clip(max=1), cutout.grid)
    timeseries = timeseries.shift(time=2) if module == "sarah" else timeseries.shift(time=1)
    return timeseries.fillna(0)

def aggregate_intra_region(timeseries, availability, **kwargs):

    region_timeseries = timeseries.where(availability > kwargs["availability_threshold"], drop=True)

    if kwargs["scale_by_availability"] & (kwargs["aggregation_method"] != "mean"):
        print("Scaling by availability is only support for method mean")

    if kwargs["aggregation_method"] == "mean":
        
        if kwargs["scale_by_availability"]:
            weighted = region_timeseries * availability
            weighted_sum = weighted.sum(dim=["lat", "lon"])
            scaling_sum = availability.sum(dim=["lat", "lon"])
            region_timeseries = weighted_sum / scaling_sum
        else:
            region_timeseries = region_timeseries.mean(dim=["lat", "lon"])

    elif kwargs["aggregation_method"] == "quantile":
        timeseries_mean = region_timeseries.mean(dim="time")

        if timeseries_mean.isnull().all():
            region_timeseries = np.zeros(region_timeseries.shape[0])
        else:
            quantile = timeseries_mean.quantile(kwargs["quantile"])
            delta = np.abs(timeseries_mean - quantile)
            lat_idx, lon_idx = np.unravel_index(delta.argmin(), delta.shape)

            region_timeseries = region_timeseries.sel(lat=timeseries_mean.lat[lat_idx], lon=timeseries_mean.lon[lon_idx])
    
    return region_timeseries


def get_nsrdb_weather_file(path,timestep,year):
    for file in os.listdir(path):
        if file.endswith(f'{timestep}_{year}.csv'):
            return file   


def find_closest_wasa_file(lat_lon_pair, file_names):
    def parse_lat_lon(file_name):
        # Extracting the lat and lon from the file name
        parts = file_name.replace('lat', '').replace('lon', '').replace('.csv', '').split('_')
        return float(parts[0]), float(parts[1])

    # Parse all lat-lon pairs from the file names
    file_lat_lons = [parse_lat_lon(name) for name in file_names]

    # Find the file with the closest lat-lon pair
    min_distance = float('inf')
    closest_file = None
    for file_name, file_lat_lon in zip(file_names, file_lat_lons):
        distance = great_circle(lat_lon_pair, file_lat_lon).kilometers
        if distance < min_distance:
            min_distance = distance
            closest_file = file_name

    return closest_file


def find_closest_nsrdb_file(lat_lon_pair, file_names):

    reippp_point = Point(lat_lon_pair)#gpd.GeoSeries([Point(lat_lon_pair)],crs="EPSG:4326").to_crs("EPSG:2049").iloc[0]
    points_list = [(f.split("_")[1], f.split("_")[0]) for f in file_names]
    points_gdf = gpd.GeoDataFrame(geometry=[Point(float(lon), float(lat)) for lat, lon in points_list], crs="EPSG:4326")#.to_crs("EPSG:2049")

    points_gdf["distance"] = points_gdf.distance(reippp_point)
    #points_gdf.to_crs("EPSG:4326", inplace=True)
    closest_point = points_gdf.loc[points_gdf["distance"].idxmin()]
    
    return closest_point.geometry.x, closest_point.geometry.y


def generate_wind_timeseries(cutout, turbine):
    wind_pu = cutout.wind(
        turbine=turbine, 
        shapes=cutout.grid, 
        smooth=False, 
        add_cutout_windspeed=True, 
        per_unit=True
    )

    return reshape_xarray(wind_pu, cutout.grid)

def load_turbine_power_curves(path):
    turbine_power_curves = pd.read_csv(path, index_col=0)

    return {
        1:{
            "hub_height": 80,
            "V":turbine_power_curves.index.values,
            "POW":turbine_power_curves["Class 1"].values,
            "P":1,     
        },

        2:{
            "hub_height": 80,
            "V":turbine_power_curves.index.values,
            "POW":turbine_power_curves["Class 2"].values,
            "P":1,     
        },

        3:{
            "hub_height": 101,
            "V":turbine_power_curves.index.values,
            "POW":turbine_power_curves["Class 3"].values,
            "P":1,     
        },

        4:{
            "hub_height": 120,
            "V":turbine_power_curves.index.values,
            "POW":turbine_power_curves["Class 3"].values,
            "P":1,     
        },

        5:{
            "hub_height": 140,
            "V":turbine_power_curves.index.values,
            "POW":turbine_power_curves["Class 3"].values,
            "P":1,     
        },
    }, turbine_power_curves

# Function to calculate the Voronoi polygons for a given supply region
def calculate_voronoi_for_supply_region(supply_regions, gdf_points, region_name):
    # Filter the supply region
    supply_region = supply_regions[supply_regions['name'] == region_name]
    points_in_region = gdf_points[gdf_points.within(supply_region.geometry.unary_union)].reset_index(drop=True)
    print(f'Number of points in {region_name}: {len(points_in_region)}')

    if points_in_region.empty:
        print(f"No points in {region_name}. Skipping Voronoi calculation.")

    # Create Voronoi polygons and clip to supply region
    multipoint = MultiPoint(points_in_region.geometry)
    voronoi_polygons = voronoi_diagram(multipoint)
    clipped_voronoi = [cell.intersection(supply_region.geometry.unary_union) for cell in voronoi_polygons.geoms if not cell.is_empty]

    cells = gpd.GeoDataFrame(geometry=clipped_voronoi)
    cells["wasa_loc"] = points_in_region["cleaned_file_name"]
    return cells

def select_intra_region_voronoi(voronoi, intra_region):
    if intra_region.crs != voronoi.crs:
        intra_region = intra_region.to_crs(voronoi.crs)
    return voronoi[voronoi.geometry.intersects(intra_region.geometry.unary_union)]

def exclude_from_voronoi(voronoi, exclusion):
    # Create a copy of the voronoi DataFrame to avoid modifying the original
    voronoi_copy = voronoi.copy()

    if exclusion.crs != voronoi_copy.crs:
        exclusion = exclusion.to_crs(voronoi_copy.crs)

    sindex = exclusion.sindex

    def difference(cell):
        possible_matches_index = list(sindex.intersection(cell.geometry.bounds))
        possible_matches = exclusion.iloc[possible_matches_index]
        precise_matches = possible_matches[possible_matches.intersects(cell.geometry)]

        cell_geometry = cell.geometry
        for _, pm in precise_matches.iterrows():
            cell_geometry = cell_geometry.difference(pm.geometry)

        return cell_geometry

    # Apply the difference function and update the geometry
    voronoi_copy['geometry'] = voronoi_copy.apply(difference, axis=1)
    return voronoi_copy

def calc_wasa_timeseries(cells, region_name, wasa_pu, method):

    cells = cells.to_crs("ESRI:54009")
    cells = cells[cells["region_name"] == region_name]
    cells["cell_area"] = cells.geometry.area
    ws = wasa_pu[cells["wasa_loc"]]

    if method == "mean":
        return ws.mean(axis=1).fillna(0)
    elif method == "area_weighted":
        cells.set_index("wasa_loc", inplace=True)
        total_area = cells["cell_area"].sum()
        return (ws.mul(cells["cell_area"], axis=1).sum(axis=1)/total_area).fillna(0)
    else:
        raise ValueError("Invalid method specified. Use mean or area_weighted.")
