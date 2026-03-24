import shutil
import os

BASE_PATH = "data"
DATASET_LIST = [
    "set_1(old_microscope_data_176_160)",
    "set_2(old_analysis_data_set_1_176_160)",
    "set_3(old_analysis_data_set_2_176_160)",
    # 'set_4(imr_90_nucleus_cytoplasm_sets_1_2_458_416)',
    # 'set_5(imr_90_nucleus_cytoplasm_cage_set_3_458_416)',
    "set_6(imr_90_cell_nucleus_cytoplasm_cage_sets_4_5_458_416)",
    "set_7(imr_90_cell_nucleus_cytoplasm_cage_set_6_458_416)",
    "set_8(jurkat_cell_cage_set_1_fs_max_2_458_320)",
    "set_9(jurkat_cell_set_1_fs_max_2_176_160)",
    "set_10(k562_cell_cage_set_1_fs_max_2_458_320)",
    "set_11(k562_cell_set_1_fs_max_2_176_160)",
    "set_12(nk92_cell_cage_set_1_fs_max_2_458_320)",
    "set_13(nk92_cell_set_1_fs_max_2_176_160)",
    "set_14(hela-suspension_cell_cage_set_1_fs_max_2_458_320)",
    "set_15(mouse-pbmc_cell_bead_cage_set_1_fs_max_2_458_320)",
    "set_16(human-pbmc_cell_set_1_fs_max_2_176_160)",
    "set 17(imr90-suspension_cell_cage_set_1_fs_max_2_458_320)",
    "set 18(imr90-suspension_cell_set_1_fs_max_2_176_160)",
]

COMBINED_DATASET_NAME = "sets_1_2_3_6_to_18"

if not os.path.exists(os.path.join(BASE_PATH, COMBINED_DATASET_NAME)):
    os.mkdir(os.path.join(BASE_PATH, COMBINED_DATASET_NAME))
if not os.path.exists(os.path.join(BASE_PATH, COMBINED_DATASET_NAME, "images")):
    os.mkdir(os.path.join(BASE_PATH, COMBINED_DATASET_NAME, "images"))
if not os.path.exists(os.path.join(BASE_PATH, COMBINED_DATASET_NAME, "images", "test")):
    os.mkdir(os.path.join(BASE_PATH, COMBINED_DATASET_NAME, "images", "test"))
if not os.path.exists(
    os.path.join(BASE_PATH, COMBINED_DATASET_NAME, "images", "train")
):
    os.mkdir(os.path.join(BASE_PATH, COMBINED_DATASET_NAME, "images", "train"))
if not os.path.exists(os.path.join(BASE_PATH, COMBINED_DATASET_NAME, "masks")):
    os.mkdir(os.path.join(BASE_PATH, COMBINED_DATASET_NAME, "masks"))
if not os.path.exists(os.path.join(BASE_PATH, COMBINED_DATASET_NAME, "masks", "test")):
    os.mkdir(os.path.join(BASE_PATH, COMBINED_DATASET_NAME, "masks", "test"))
if not os.path.exists(os.path.join(BASE_PATH, COMBINED_DATASET_NAME, "masks", "train")):
    os.mkdir(os.path.join(BASE_PATH, COMBINED_DATASET_NAME, "masks", "train"))

all_images = {}
images_num = {}

for dataset_type in ["test", "train"]:
    all_images[dataset_type] = []
    for dataset_name in DATASET_LIST:
        all_images[dataset_type] += os.listdir(
            os.path.join(BASE_PATH, dataset_name, "images", dataset_type)
        )

    all_images[dataset_type] = [
        filename
        for filename in all_images[dataset_type]
        if filename[-3:].lower() == "jpg" or filename[-4:].lower() == "jpeg"
    ]

    images_num[dataset_type] = {}
    for filename in all_images[dataset_type]:
        if filename in images_num[dataset_type]:
            images_num[dataset_type][filename] += 1
        else:
            images_num[dataset_type][filename] = 1


for dataset_type in ["test", "train"]:
    for dataset_name in DATASET_LIST:
        source_dir_images = os.path.join(
            BASE_PATH, dataset_name, "images", dataset_type
        )
        source_dir_masks = os.path.join(BASE_PATH, dataset_name, "masks", dataset_type)
        dest_dir_images = os.path.join(
            BASE_PATH, COMBINED_DATASET_NAME, "images", dataset_type
        )
        dest_dir_masks = os.path.join(
            BASE_PATH, COMBINED_DATASET_NAME, "masks", dataset_type
        )

        images_to_copy = os.listdir(source_dir_images)
        images_to_copy = [
            filename
            for filename in images_to_copy
            if filename[-3:].lower() == "jpg" or filename[-4:].lower() == "jpeg"
        ]
        for filename in images_to_copy:
            name = ".".join(filename.strip().split(".")[:-1])
            if images_num[dataset_type][filename] == 1:
                shutil.copy(
                    os.path.join(source_dir_images, filename),
                    os.path.join(dest_dir_images, filename),
                )
                shutil.copy(
                    os.path.join(source_dir_masks, name + ".pkl"),
                    os.path.join(dest_dir_masks, name + ".pkl"),
                )
            else:
                name_ext = "_" + str(images_num[dataset_type][filename])
                images_num[dataset_type][filename] -= 1
                shutil.copy(
                    os.path.join(source_dir_images, filename),
                    os.path.join(dest_dir_images, name + name_ext + ".jpg"),
                )
                shutil.copy(
                    os.path.join(source_dir_masks, name + ".pkl"),
                    os.path.join(dest_dir_masks, name + name_ext + ".pkl"),
                )
