import time

import clearml
from clearml import Task, Dataset
import numpy as np
import geopandas as gpd
import pandas as pd
import rasterio
from shapely.geometry import Point, LineString, Polygon, MultiPoint, MultiLineString, MultiPolygon, box
import shapely.vectorized
import os

def create_mask(poly, x_coords, y_coords):
    '''Returns mask created from Polygon `poly` and coordinates `x_coords`, `y_coords`
    that would normally be the output of `shapely.vectorized.contains(poly, x_coords, y_coords)`
    but is also able to handle any potential OverflowErrors'''

    def handle_overflow(p, x, y):
        '''Returns Shapely.contains mask as intended if no OverflowError is detected,
        otherwise returns the OverflowError exception that would have been raised'''
        try: mask = shapely.vectorized.contains(p, x, y)
        except OverflowError as e:
            return e
        return mask

    mask_final = handle_overflow(poly, x_coords, y_coords)
    while isinstance(mask_final, OverflowError):
        # recursively subdivide the raster band into smaller chunks, then re-combine them
        if len(x_coords) < len(y_coords):
            split = len(y_coords) // 2
            mask_left = create_mask(poly, x_coords[:, :split], y_coords[:, :split])
            mask_right = create_mask(poly, x_coords[:, split:], y_coords[:, split:])
            mask_final = np.hstack((mask_left, mask_right))
        else:
            split = len(x_coords) // 2
            mask_upper = create_mask(poly, x_coords[:split, :], y_coords[:split, :])
            mask_lower = create_mask(poly, x_coords[split:, :], y_coords[split:, :])
            mask_final = np.vstack((mask_upper, mask_lower))
    return mask_final

dataset_name = 'Ethnologue Population Mapping'

# REMOVED: create/upload datasets on ClearML
# dataset = Dataset.create(
#     dataset_project="Ethnologue_Richard_Internship", dataset_name=dataset_name
# )
# num_links = dataset.add_files(path="./Language Polygons/SIL_lang_polys_June2022/", dataset_path="/Language Polygons/")
# num_links += dataset.add_files(path="./population_af_2018-10-01/", dataset_path="/Facebook Dataset/")
# dataset.upload()
# print(f"Dataset '{dataset_name}' generated, with {num_links} files added.")
# dataset.finalize()

# prepare task on ClearML
Task.add_requirements("-rrequirements.txt")
task = Task.init(
  project_name='Ethnologue_Richard_Internship',    # project name of at least 3 characters
  task_name='Uganda' + str(int(time.time())), # task name of at least 3 characters
  task_type="training",
  tags=None,
  reuse_last_task_id=True,
  continue_last_task=False,
  output_uri="s3://richard-ethnologue-gis",
  auto_connect_arg_parser=True,
  auto_connect_frameworks=True,
  auto_resource_monitoring=True,
  auto_connect_streams=True,
)

dataset_path = Dataset.get(dataset_project='Ethnologue_Richard_Internship', dataset_name=dataset_name).get_local_copy()
fp = os.path.join(dataset_path, "Language Polygons/SIL_lang_polys_June2022.shp")
data = gpd.read_file(fp)

### CHANGE COUNTRY LABELS HERE ###
ctry_name = "Uganda"
ctry_abbr = "UGA"

grouped = data.groupby("COUNTRY_IS")
ctry = grouped.get_group(ctry_abbr, data)
ctry["Population"] = 0

# move to country's file directory (or create new folder for country if not already existing)
ctry_dir = os.path.join("Countries", ctry_name)
if not os.path.exists(ctry_dir):
    os.makedirs(ctry_dir)
os.chdir(ctry_dir)

population_data = os.path.join(dataset_path, "Facebook Dataset")
tifs = list() # get only the tif files from population data
for file in os.listdir(population_data):
    if file[-4:] == ".tif":
        tifs.append(file)

outfp = f"{ctry_name}_Population_Estimates.shp"
not_opened = tifs[:]

if os.path.exists("Population_files_not_opened.txt"):
    ctry = gpd.read_file(outfp) # in case kernel crashes, keep track of progress
    with open("Population_files_not_opened.txt",'r') as f:
        not_opened = f.read().splitlines()

while len(not_opened) > 0:
    file = not_opened.pop()
    src = rasterio.open(os.path.join(population_data, file), "r")
    print(f"Opening {file}")
    #print(src.meta)
    transformer = src.meta['transform']
    width = src.meta['width']
    height = src.meta['height']

    # TODO: get location of raster data and check if it overlaps with any of the language polygons
    overlap_polys = list()
    region = box(*src.bounds)

    for i in ctry.index.values:
        poly = ctry.loc[i, "geometry"]
        if type(poly) == Polygon:
            poly_list = [poly]
        elif type(poly) == MultiPolygon:
            poly_list = list(poly)

        for p in poly_list:
            checker = MultiPolygon([p, region])
            if not checker.is_valid:
                overlap_polys.append((i,p))

    # If yes: create masks from language polygons and sum over src pixel values with label True
    if len(overlap_polys) > 0:
        print(f"{file} overlaps with language polys!")
        # multi_poly = MultiPolygon(list(zip(*overlap_polys))[1])

        coords = np.indices((width, height)) # image coordinates
        print("Transforming the coordinates...")
        x,y = transformer * coords # transform to geo coords (this step is REALLY slow!)
        print("Reading the raster data...")
        vals = src.read(1) # 2D population raster data
        print("Getting the population counts...")

        allow_overcounts = True # boolean parameter to allow double-counting for overlapping polygons, i.e. each person counts as 1 for every polygon
                                # If set to False, then for each pixel where >1 polygons overlap, its value will be equally distributed among those polygons
        if not allow_overcounts:
            for i,p in overlap_polys:
                mask = create_mask(p, x, y)
                print(mask.shape, vals.shape)
                #pop_count = vals.transpose() * mask # element-wise multiplication
                pop_count = np.extract(mask, vals.transpose())
                ctry.loc[i, "Population"] += np.nansum(pop_count)
                print("Yay, I finished one polygon!")

        else:
            equalizer_inv = np.zeros((width, height))
            poly_masks = list()
            for i,p in overlap_polys:
                # if raster band is too large, split in half and re-merge (repeat until no error raised)
                mask = create_mask(p, x, y)
                print(mask.shape, vals.shape)
                equalizer_inv += mask
                poly_masks.append((i,mask))
            equalizer_inv[equalizer_inv == 0] = np.inf
            for i,mask in poly_masks:
                pop_count = 1/equalizer_inv * mask * vals.transpose()
                ctry.loc[i,"Population"] += np.nansum(pop_count)
                print("Yay, I finished one polygon!")

        print("All overlapping polygons have been successfully parsed.")
        # update files with new counts
        ctry.to_file(outfp)
        with open("Population_files_not_opened.txt", 'w') as f:
            f.write("\n".join(not_opened))

# reaching end of while loop means all population files were parsed
os.remove("Population_files_not_opened.txt")

# write results to csv file
result = ctry[["ETH_LG_R", "ETH_NO", "ISO_LANGUA", "COUNTRY_IS", "Population"]]
result.rename(columns={"ISO_LANGUA": "ISO_639",
                       "COUNTRY_IS": "COUNTRY"})
# result.to_csv(f"{ctry_name}_Population_Estimates.csv")

# Save the artifact in ClearML
task.upload_artifact(name='Uganda', artifact_object=result)
