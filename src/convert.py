import os
import shutil
from collections import defaultdict
from urllib.parse import unquote, urlparse

import numpy as np
import pycocotools.mask as mask_util
import supervisely as sly
from dataset_tools.convert import unpack_if_archive
from PIL import Image
from supervisely.io.fs import (
    file_exists,
    get_file_ext,
    get_file_name,
    get_file_name_with_ext,
)
from supervisely.io.json import load_json_file
from tqdm import tqdm

Image.MAX_IMAGE_PIXELS = None

import src.settings as s


def download_dataset(teamfiles_dir: str) -> str:
    """Use it for large datasets to convert them on the instance"""
    api = sly.Api.from_env()
    team_id = sly.env.team_id()
    storage_dir = sly.app.get_data_dir()

    if isinstance(s.DOWNLOAD_ORIGINAL_URL, str):
        parsed_url = urlparse(s.DOWNLOAD_ORIGINAL_URL)
        file_name_with_ext = os.path.basename(parsed_url.path)
        file_name_with_ext = unquote(file_name_with_ext)

        sly.logger.info(f"Start unpacking archive '{file_name_with_ext}'...")
        local_path = os.path.join(storage_dir, file_name_with_ext)
        teamfiles_path = os.path.join(teamfiles_dir, file_name_with_ext)

        fsize = api.file.get_directory_size(team_id, teamfiles_dir)
        with tqdm(
            desc=f"Downloading '{file_name_with_ext}' to buffer...",
            total=fsize,
            unit="B",
            unit_scale=True,
        ) as pbar:
            api.file.download(team_id, teamfiles_path, local_path, progress_cb=pbar)
        dataset_path = unpack_if_archive(local_path)

    if isinstance(s.DOWNLOAD_ORIGINAL_URL, dict):
        for file_name_with_ext, url in s.DOWNLOAD_ORIGINAL_URL.items():
            local_path = os.path.join(storage_dir, file_name_with_ext)
            teamfiles_path = os.path.join(teamfiles_dir, file_name_with_ext)

            if not os.path.exists(get_file_name(local_path)):
                fsize = api.file.get_directory_size(team_id, teamfiles_dir)
                with tqdm(
                    desc=f"Downloading '{file_name_with_ext}' to buffer...",
                    total=fsize,
                    unit="B",
                    unit_scale=True,
                ) as pbar:
                    api.file.download(team_id, teamfiles_path, local_path, progress_cb=pbar)

                sly.logger.info(f"Start unpacking archive '{file_name_with_ext}'...")
                unpack_if_archive(local_path)
            else:
                sly.logger.info(
                    f"Archive '{file_name_with_ext}' was already unpacked to '{os.path.join(storage_dir, get_file_name(file_name_with_ext))}'. Skipping..."
                )

        dataset_path = storage_dir
    return dataset_path


def count_files(path, extension):
    count = 0
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(extension):
                count += 1
    return count


def convert_and_upload_supervisely_project(
    api: sly.Api, workspace_id: int, project_name: str
) -> sly.ProjectInfo:
    ### Function should read local dataset and upload it to Supervisely project, then return project info.###
    images_path = "/home/alex/DATASETS/TODO/EntitySeg"  # subfolders must have names entity_01_11580, entity_02_11598, entity_03_10049

    train_ann_path = "/home/alex/DATASETS/TODO/EntitySeg/entityseg_train.json"
    val_ann_path = "/home/alex/DATASETS/TODO/EntitySeg/entityseg_val.json"

    batch_size = 10

    ds_name_to_split = {"train": train_ann_path, "val": val_ann_path}

    def convert_rle_mask_to_polygon(rle_mask_data):
        if type(rle_mask_data["counts"]) is str:
            rle_mask_data["counts"] = bytes(rle_mask_data["counts"], encoding="utf-8")
            mask = mask_util.decode(rle_mask_data)
        else:
            rle_obj = mask_util.frPyObjects(
                rle_mask_data,
                rle_mask_data["size"][0],
                rle_mask_data["size"][1],
            )
            mask = mask_util.decode(rle_obj)
        mask = np.array(mask, dtype=bool)
        if np.any(mask) == [0]:
            return None
        return sly.Bitmap(mask).to_contours()

    def create_ann(image_path):
        labels = []

        subfolder_value = image_path.split("/")[-2]
        subfolder = sly.Tag(subfolder_meta, value=subfolder_value)

        image = Image.open(image_path)
        img_height = image.height
        img_wight = image.width

        ann_data = image_name_to_ann_data[get_file_name_with_ext(image_path)]
        for curr_ann_data in ann_data:
            tags = []
            category_id = curr_ann_data[0]
            obj_class = idx_to_obj_class[category_id]
            supercategory_value = id_to_supercategory.get(category_id)
            supercategory = sly.Tag(supercategory_meta, value=supercategory_value)
            tags.append(supercategory)

            type_value = id_to_type.get(category_id)
            type_tag = sly.Tag(type_meta, value=type_value)
            tags.append(type_tag)

            rle_mask_data = curr_ann_data[1]
            polygons = convert_rle_mask_to_polygon(rle_mask_data)
            if polygons is not None:
                for polygon in polygons:
                    label = sly.Label(polygon, obj_class, tags=tags)
                    labels.append(label)

            bbox_coord = curr_ann_data[2]
            rectangle = sly.Rectangle(
                top=int(bbox_coord[1]),
                left=int(bbox_coord[0]),
                bottom=int(bbox_coord[1] + bbox_coord[3]),
                right=int(bbox_coord[0] + bbox_coord[2]),
            )
            label_rectangle = sly.Label(rectangle, obj_class, tags=tags)
            labels.append(label_rectangle)

        return sly.Annotation(img_size=(img_height, img_wight), labels=labels, img_tags=[subfolder])

    project = api.project.create(workspace_id, project_name, change_name_if_conflict=True)

    supercategory_meta = sly.TagMeta("supercategory", sly.TagValueType.ANY_STRING)
    type_meta = sly.TagMeta("type", sly.TagValueType.ANY_STRING)
    subfolder_meta = sly.TagMeta("subfolder", sly.TagValueType.ANY_STRING)

    meta = sly.ProjectMeta(tag_metas=[supercategory_meta, type_meta, subfolder_meta])

    idx_to_obj_class = {}

    for ds_name, ann_path in ds_name_to_split.items():
        dataset = api.dataset.create(project.id, ds_name, change_name_if_conflict=True)

        image_id_to_name = {}
        image_name_to_ann_data = defaultdict(list)
        id_to_supercategory = {}
        id_to_type = {}
        all_images_names = []
        test = {}

        ann = load_json_file(ann_path)
        for curr_category in ann["categories"]:
            if meta.get_obj_class(curr_category["name"].strip()) is None:
                obj_class = sly.ObjClass(curr_category["name"].strip(), sly.AnyGeometry)
                meta = meta.add_obj_class(obj_class)
            id_to_supercategory[curr_category["id"]] = curr_category["supercategory"]
            id_to_type[curr_category["id"]] = curr_category["type"]
            idx_to_obj_class[curr_category["id"]] = obj_class
        api.project.update_meta(project.id, meta.to_json())

        for curr_image_info in ann["images"]:
            image_id_to_name[curr_image_info["id"]] = curr_image_info["file_name"]

        for curr_ann_data in ann["annotations"]:
            image_id = curr_ann_data["image_id"]
            all_images_names.append(image_id_to_name[image_id])
            image_name_to_ann_data[image_id_to_name[image_id].split("/")[1]].append(
                [curr_ann_data["category_id"], curr_ann_data["segmentation"], curr_ann_data["bbox"]]
            )

        images_names = list(set(all_images_names))

        progress = sly.Progress("Create dataset {}".format(ds_name), len(images_names))

        for img_names_batch in sly.batched(images_names, batch_size=batch_size):
            images_names_batch = []
            img_pathes_batch = []
            for im_name in img_names_batch:
                img_pathes_batch.append(os.path.join(images_path, im_name))
                images_names_batch.append(im_name.split("/")[1])

            img_infos = api.image.upload_paths(dataset.id, img_names_batch, img_pathes_batch)
            img_ids = [im_info.id for im_info in img_infos]

            anns = [create_ann(image_path) for image_path in img_pathes_batch]
            api.annotation.upload_anns(img_ids, anns)

            progress.iters_done_report(len(img_names_batch))

        return project
