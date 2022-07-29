from clearml import Task, Dataset
import random
import numpy as np
import geopandas as gpd
import pandas as pd
import rasterio
from shapely.geometry import Polygon, MultiPolygon, box
import shapely.vectorized
import os
import gc

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
            del mask_left
            del mask_right
        else:
            split = len(x_coords) // 2
            mask_upper = create_mask(poly, x_coords[:split, :], y_coords[:split, :])
            mask_lower = create_mask(poly, x_coords[split:, :], y_coords[split:, :])
            mask_final = np.vstack((mask_upper, mask_lower))
            del mask_upper
            del mask_lower
        gc.collect()
    return mask_final

def process(filename, poly_list, file_dir='', allow_overcounts=True):
    '''Receives geotiff file as input and returns the sum of values for pixels that overlap with polygons from `poly_list`

    allow_overcounts: boolean parameter to allow double-counting for overlapping polygons, i.e. each person counts as 1 for every polygon
    If set to False, then for each pixel where >1 polygons overlap, its value will be equally distributed among those polygons'''

    src = rasterio.open(os.path.join(file_dir, filename), "r")
    print(f"Opening {filename}")
    # print(src.meta)
    transformer = src.meta['transform']
    width = src.meta['width']
    height = src.meta['height']

    # TODO: get location of raster data and check if it overlaps with any of the language polygons
    overlap_polys = list()
    region = box(*src.bounds)

    for i, poly in poly_list:
        if type(poly) == Polygon:
            sub_polys = [poly]
        elif type(poly) == MultiPolygon:
            sub_polys = list(poly)

        for p in sub_polys:
            checker = MultiPolygon([p, region])
            if not checker.is_valid:
                overlap_polys.append((i,p))

    # If yes: create masks from language polygons and sum over src pixel values with label True
    if len(overlap_polys) > 0:
        print(f"{filename} overlaps with language polys!")
        results = list()

        coords = np.indices((width, height)) # image coordinates
        print("Transforming the coordinates...")
        x,y = transformer * coords # transform to geo coords
        print("Reading the raster data...")
        vals = src.read(1) # 2D population raster data
        print("Getting the population counts...")

        if allow_overcounts:
            for i,p in overlap_polys:
                mask = create_mask(p, x, y)
                print(mask.shape, vals.shape)
                #pop_count = vals.transpose() * mask # element-wise multiplication
                pop_count = np.extract(mask, vals.transpose())
                results.append((i, pop_count))
                print("Yay, I finished one polygon!")
                del mask
                gc.collect()

        else:
            equalizer_inv = np.zeros((width, height))
            poly_masks = list()
            for i,p in overlap_polys:
                # if raster band is too large, split in half and re-merge (repeat until no error raised)
                mask = create_mask(p, x, y)
                print(mask.shape, vals.shape)
                equalizer_inv += mask
                poly_masks.append((i,mask))
                del mask
                gc.collect()
            equalizer_inv[equalizer_inv == 0] = np.inf
            equalizer = 1/equalizer_inv
            del equalizer_inv
            gc.collect()
            for i,mask in poly_masks:
                pop_count = equalizer * mask * vals.transpose()
                results.append((i, pop_count))
                print("Yay, I finished one polygon!")

        print("All overlapping polygons have been successfully parsed.")
        return results

### CHANGE COUNTRY LABELS HERE ###
ctry_name = "Uganda" # full name of country
ctry_abbr = "UGA" # ISO-3 code of country

### MAIN METHOD ###
def main():
    dataset_project = 'Ethnologue_Richard_Internship'
    dataset_name = 'Ethnologue Population Mapping'

    # REMOVED: create/upload dataset onto ClearML
    # dataset = Dataset.create(
    #     dataset_project=dataset_project, dataset_name=dataset_name
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
      task_name=f'{ctry_name}_Population_Estimates', # task name of at least 3 characters
      task_type="data_processing",
      tags=None,
      reuse_last_task_id=True,
      continue_last_task=True,
      output_uri="s3://richard-ethnologue-gis",
      auto_connect_arg_parser=True,
      auto_connect_frameworks=True,
      auto_resource_monitoring=True,
      auto_connect_streams=True,
      )
    task.set_base_docker(docker_image="python:3.9.7")

    # get local copy of population dataset
    dataset_path = Dataset.get(dataset_project=dataset_project, dataset_name=dataset_name).get_local_copy()
    population_data = os.path.join(dataset_path, "Facebook Dataset")

    # check if there is a previous artifact registered
    prev_artifact = task.get_registered_artifacts()
    if len(prev_artifact) > 0:
        ctry = prev_artifact[f'{ctry_name}']
        unopened_files = task.artifacts["unopened_files"]

    # otherwise, initialize new `ctry` dataframe
    else:
        # import language polygon data
        fp = os.path.join(dataset_path, "Language Polygons/SIL_lang_polys_June2022.shp")
        data = gpd.read_file(fp)

        # extract only language polygons of the desired country
        grouped = data.groupby("COUNTRY_IS")
        ctry = grouped.get_group(ctry_abbr, data)

        # keep only the relevant columns
        ctry = ctry[["ETH_LG_R", "ETH_NO", "ISO_LANGUA", "COUNTRY_IS", "geometry"]]
        ctry.rename(columns={"ISO_LANGUA": "ISO_639",
                             "COUNTRY_IS": "COUNTRY"})
        ctry["Population"] = 0 # add new column to store population estimates
        task.register_artifact(f'{ctry_name}', ctry)

        unopened_files = [file for file in os.listdir(population_data) if file[-4:] == ".tif"]
        random.shuffle(unopened_files)
        del data
        del grouped
        gc.collect()

    try: # get list of language polygons from country data, with index included
        poly_list = list(ctry["geometry"].items())
    except KeyError: # another way to check that process is completed is to see whether len(unopened_files) == 0
        print("All files from population density data have been processed.")

    # process the data to generate population estimates for each language
    for file in unopened_files:
        results = process(file, poly_list, file_dir=population_data, allow_overcounts=False)
        # Save progress in ClearML
        if results is not None:
            for i, pop_count in results:
                ctry.loc[i, "Population"] += np.nansum(pop_count)
        unopened_files.remove(file)
        task.upload_artifact(name="unopened_files", artifact_object=unopened_files)

    # If all .tif files have been processed, remove `geometry` column from `ctry` dataframe
    ctry.drop("geometry", axis=1, inplace=True)

if __name__ == '__main__':
    main()
