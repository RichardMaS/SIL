from argparse import ArgumentParser
from clearml import Task, Dataset

import sys
import random
import pickle
import os
import gc

import numpy as np
import geopandas as gpd
import pandas as pd
import rasterio
from shapely.geometry import Polygon, MultiPolygon, box
import shapely.vectorized
import pycountry

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

def process(filename, poly_list, df, file_dir='', allow_overcounts=True, memory_limit=100):
    '''Receives geotiff file as input and updates the population counts in `df` for any overlaps with polygons from `poly_list`

    allow_overcounts (bool): allow double-counting for overlapping polygons, i.e. each person counts as 1 for every polygon
    If set to False, then for each pixel where >1 polygons overlap, its value will be equally distributed among those polygons

    memory_limit (int): max number of polygons whose regional masks can be stored internally to avoid allocating too much memory, default 100'''

    src = rasterio.open(os.path.join(file_dir, filename), "r")
    print(f"Opening {filename}")
    # print(src.meta)
    transformer = src.meta['transform']
    width = src.meta['width']
    height = src.meta['height']

    # get location of raster data and check if it overlaps with any of the language polygons
    overlap_polys = list()
    region = box(*src.bounds)

    for i, poly in poly_list:
        if type(poly) == Polygon:
            sub_polys = [poly]
        elif type(poly) == MultiPolygon:
            sub_polys = list(poly)

        for p in sub_polys:
            if not MultiPolygon([p, region]).is_valid:
                overlap_polys.append((i,p))

    # If yes: create masks from language polygons and sum over src pixel values with label True
    if len(overlap_polys) > 0:
        print(f"{filename} overlaps with language polys!")

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
                df.loc[i, "Population"] += np.nansum(pop_count)
                print("Yay, I finished one polygon!")
                del mask
                gc.collect()

        else:
            equalizer_inv = np.zeros((width, height))
            poly_masks = list()
            for i,p in overlap_polys:
                # if raster band is too large, split in half and re-merge (repeat until no error raised)
                mask = create_mask(p, x, y)
                equalizer_inv += mask
                if not len(overlap_polys) > memory_limit: # cap number of masks stored in list to prevent process from running out of memory
                    print(mask.shape, vals.shape)
                    poly_masks.append((i,mask))
                del mask
                gc.collect()
            equalizer_inv[equalizer_inv == 0] = np.inf
            equalizer = 1/equalizer_inv
            del equalizer_inv
            gc.collect()
            for j in range(len(overlap_polys)):
                if len(overlap_polys) > memory_limit:
                    i,p = overlap_polys[j]
                    mask = create_mask(p, x, y)
                    print(mask.shape, vals.shape)
                else: i,mask = poly_masks[j]
                pop_count = equalizer * mask * vals.transpose()
                df.loc[i, "Population"] += np.nansum(pop_count)
                print("Yay, I finished one polygon!")

        print("All overlapping polygons have been successfully parsed.")

### MAIN METHOD ###
def main():
    ctry_abbr = Task.current_task().get_parameter("Args/ctry_abbr") # recover command-line arg from hyperparams
    ctry_name = pycountry.countries.get(alpha_3=ctry_abbr).name

    # get local copy of population dataset
    dataset_path = Dataset.get(dataset_project=project_name, dataset_name=dataset_name).get_local_copy()
    population_data = os.path.join(dataset_path, "Facebook Dataset")

    # import language polygon data
    fp = os.path.join(dataset_path, "Language Polygons/SIL_lang_polys_June2022.shp")
    data = gpd.read_file(fp)

    # extract only language polygons of the desired country
    grouped = data.groupby("COUNTRY_IS")
    ctry = grouped.get_group(ctry_abbr, data)

    try: # check if there is a previously run task for which to continue progress
        prev_task = Task.get_tasks(project_name=project_name, task_name=task_name,
                                   task_filter={'order_by': ['-last_update']})[1]
        results_url = prev_task.artifacts[f'{ctry_name}'].get_local_copy()
        files_url = prev_task.artifacts["unopened_files"].get_local_copy()
        results = pd.read_csv(results_url, compression='gzip')

        # continue executing process on remaining .tif files if any exist
        unopened_files = pickle.load(open(files_url, "rb"), encoding='latin1')
        print(f"Picking up from {len(unopened_files)} unopened file(s)")

    # if no previous task or no artifacts found, initialize new `results` dataframe
    except (IndexError, KeyError, IsADirectoryError):
        # keep only the relevant columns
        results = ctry[["ETH_LG_R", "ETH_NO", "ISO_LANGUA", "COUNTRY_IS"]]
        results.rename(columns={"ISO_LANGUA": "ISO_639", "COUNTRY_IS": "COUNTRY"}, inplace=True)
        results["Population"] = 0 # add new column to store population estimates

        # initialize list of .tif files from population data in randomized order
        unopened_files = [file for file in os.listdir(population_data) if file[-4:] == ".tif"]
        random.shuffle(unopened_files)

    task.register_artifact(name=f'{ctry_name}', artifact=results)
    poly_list = list(ctry["geometry"].items()) # get list of language polygons from country data, with index included

    del data
    del grouped
    del ctry
    gc.collect()

    # process the data to generate population estimates for each language
    while len(unopened_files) > 0:
        file = unopened_files.pop()
        process(file, poly_list, results, file_dir=population_data, allow_overcounts=False)
        # Save progress in ClearML
        task.upload_artifact(name="unopened_files", artifact_object=unopened_files)


if __name__ == '__main__':
    project_name = 'Ethnologue_Richard_Internship' # project name of at least 3 characters
    dataset_name = 'Ethnologue Population Mapping' # dataset name of at least 3 characters
    task_name = 'Population_Estimator' # task name of at least 3 characters

    # parse ISO alpha-3 country code as command line argument
    parser = ArgumentParser()
    parser.add_argument("ctry_abbr", type=str, metavar="ISO-alpha-3-country-code")
    args = parser.parse_args()

    # REMOVED: create/upload dataset onto ClearML
    # dataset = Dataset.create(
    #     dataset_project=project_name, dataset_name=dataset_name
    # )
    # num_links = dataset.add_files(path="./Language Polygons/SIL_lang_polys_June2022/", dataset_path="/Language Polygons/")
    # num_links += dataset.add_files(path="./population_af_2018-10-01/", dataset_path="/Facebook Dataset/")
    # dataset.upload()
    # print(f"Dataset '{dataset_name}' generated, with {num_links} files added.")
    # dataset.finalize()

    # prepare task on ClearML
    Task.add_requirements("-rrequirements.txt")
    task = Task.init(
      project_name=project_name,
      task_name=task_name,
      task_type="data_processing",
      tags=None,
      reuse_last_task_id=False,
      continue_last_task=False,
      output_uri="s3://richard-ethnologue-gis",
      auto_connect_arg_parser=True,
      auto_connect_frameworks=True,
      auto_resource_monitoring=True,
      auto_connect_streams=True,
      )
    task.set_base_docker(docker_image="python:3.9.7")
    task.execute_remotely(queue_name="idx_10gb")
    main()
