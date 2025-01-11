import os
import shutil
from tqdm import tqdm
import numpy as np
from numpy  import array
import SimpleITK as sitk
from PIL import Image
from collections import OrderedDict
from batchgenerators.utilities.file_and_folder_operations import *

def read_pkl(path):
    fp = open(path, "rb")
    return pickle.load(fp)

def write_pkl(data, f):
    pickle.dump(data, f, -1)


def copy_BraTS_segmentation_and_convert_labels(in_file, out_file, ped=False):
    # use this for segmentation only!!!
    # nnUNet wants the labels to be continuous. BraTS is 0, 1, 2, 4 -> we make that into 0, 1, 2, 3
    img = sitk.ReadImage(in_file)
    img_npy = sitk.GetArrayFromImage(img)

    uniques = np.unique(img_npy)
    for u in uniques:
        if u not in [0, 1, 2, 4]:
            raise RuntimeError('unexpected label')

    if ped:
        seg_new = np.zeros_like(img_npy)
        seg_new[img_npy != 0] = -1
        img_corr = sitk.GetImageFromArray(seg_new)
        img_corr.CopyInformation(img)
        sitk.WriteImage(img_corr, out_file)

    else:
        seg_new = np.zeros_like(img_npy)
        seg_new[img_npy == 4] = 3
        seg_new[img_npy == 2] = 1
        seg_new[img_npy == 1] = 2
        img_corr = sitk.GetImageFromArray(seg_new)
        img_corr.CopyInformation(img)
        sitk.WriteImage(img_corr, out_file)

def merge_splits_final_funtion(splits_final_path_list, output_path):
    splits_final_list = [read_pkl(splits_final_path)[0] for splits_final_path in splits_final_path_list]
    merge_splits_final = OrderedDict()
    merge_splits_final["train"] = []
    for sub_splits_final in splits_final_list:
        merge_splits_final["train"].extend(sub_splits_final["train"])

    merge_splits_final["val"] = []
    for sub_splits_final in splits_final_list:
        merge_splits_final["val"].extend(sub_splits_final["val"])

    f = open(os.path.join(output_path, "splits_final.pkl"), "wb")
    write_pkl([merge_splits_final], f)

    f.close()

def merge_dataset_properties_funtion(dataset_properties_path_list, output_path):
    dataset_properties_list = [read_pkl(dataset_properties_path) for dataset_properties_path in dataset_properties_path_list]
    merge_dataset_properties = {}
    # keys = ['all_sizes', 'all_spacings', 'all_classes', 'modalities', 'intensityproperties', 'size_reductions']
    #all_sizes
    merge_dataset_properties["all_sizes"] = []
    for sub_properties in dataset_properties_list:
        merge_dataset_properties["all_sizes"].extend(sub_properties["all_sizes"])

    # all_spacings
    merge_dataset_properties["all_spacings"] = []
    for sub_properties in dataset_properties_list:
        merge_dataset_properties["all_spacings"].extend(sub_properties["all_spacings"])

    #all_classes
    merge_dataset_properties["all_classes"] = []
    for sub_properties in dataset_properties_list:
        class_list = sub_properties["all_classes"]
        for sub_class in class_list:
            if sub_class not in merge_dataset_properties["all_classes"]:
                merge_dataset_properties["all_classes"].append(sub_class)
    merge_dataset_properties["all_classes"] = sorted(merge_dataset_properties["all_classes"])

    #modalities
    merge_dataset_properties["modalities"] = {}
    for sub_properties in dataset_properties_list:
        class_dic = sub_properties["modalities"]
        for id, modality in class_dic.items():
            if id not in merge_dataset_properties["modalities"]:
                merge_dataset_properties["modalities"][id] = [modality]
            else:
                if modality not in merge_dataset_properties["modalities"][id]:
                    merge_dataset_properties["modalities"][id].append(modality)

    #intensityproperties
    merge_dataset_properties["intensityproperties"] = None

    #size_reductions
    merge_dataset_properties["size_reductions"] = OrderedDict()
    for sub_properties in dataset_properties_list:
        for key, value in sub_properties["size_reductions"].items():
            merge_dataset_properties["size_reductions"][key] = value

    f = open(os.path.join(output_path, "dataset_properties.pkl"), "wb")
    write_pkl(merge_dataset_properties, f)

    f.close()



def merge_DoDNetPlans_plans_3D_funtion(DoDNetPlans_plans_3D_path_list, output_path):
    DoDNetPlans_plans_3D_list = [read_pkl(DoDNetPlans_plans_3D_path) for DoDNetPlans_plans_3D_path in DoDNetPlans_plans_3D_path_list]
    merge_DoDNetPlans_plans_3D = {}
    # ['num_stages', 'num_modalities', 'modalities', 'normalization_schemes', 'dataset_properties', 'list_of_npz_files',
    #  'original_spacings', 'original_sizes', 'preprocessed_data_folder', 'num_classes', 'all_classes',
    #  'base_num_features', 'use_mask_for_norm', 'keep_only_largest_region', 'min_region_size_per_class',
    #  'min_size_per_class', 'transpose_forward', 'transpose_backward', 'data_identifier', 'plans_per_stage',
     # 'preprocessor_name', 'conv_per_stage']
    #num_stages
    merge_DoDNetPlans_plans_3D["num_stages"] = 1

    #num_modalities
    max_modalities = 0
    for sub_DoDNetPlans_plans_3D in DoDNetPlans_plans_3D_list:
        sub_modalities = sub_DoDNetPlans_plans_3D["num_modalities"]
        if sub_modalities > max_modalities:
            max_modalities = sub_modalities
    merge_DoDNetPlans_plans_3D["num_modalities"] = max_modalities

    #modalities
    merge_DoDNetPlans_plans_3D["modalities"] = {}
    for sub_DoDNetPlans_plans_3D in DoDNetPlans_plans_3D_list:
        class_dic = sub_DoDNetPlans_plans_3D["modalities"]
        for id, modality in class_dic.items():
            if id not in merge_DoDNetPlans_plans_3D["modalities"]:
                merge_DoDNetPlans_plans_3D["modalities"][id] = [modality]
            else:
                if modality not in merge_DoDNetPlans_plans_3D["modalities"][id]:
                    merge_DoDNetPlans_plans_3D["modalities"][id].append(modality)

    #normalization_schemes
    merge_DoDNetPlans_plans_3D["normalization_schemes"] = None

    #dataset_properties
    merge_DoDNetPlans_plans_3D["dataset_properties"] = read_pkl(os.path.join(output_path, "dataset_properties.pkl"))

    #list_of_npz_files
    merge_DoDNetPlans_plans_3D["list_of_npz_files"] = []
    for sub_DoDNetPlans_plans_3D in DoDNetPlans_plans_3D_list:
        merge_DoDNetPlans_plans_3D["list_of_npz_files"].extend(sub_DoDNetPlans_plans_3D["list_of_npz_files"])

    #original_spacings
    merge_DoDNetPlans_plans_3D["original_spacings"] = []
    for sub_DoDNetPlans_plans_3D in DoDNetPlans_plans_3D_list:
        merge_DoDNetPlans_plans_3D["original_spacings"].extend(sub_DoDNetPlans_plans_3D["original_spacings"])

    #original_sizes
    merge_DoDNetPlans_plans_3D["original_sizes"] = []
    for sub_DoDNetPlans_plans_3D in DoDNetPlans_plans_3D_list:
        merge_DoDNetPlans_plans_3D["original_sizes"].extend(sub_DoDNetPlans_plans_3D["original_sizes"])

    #preprocessed_data_folder
    merge_DoDNetPlans_plans_3D["preprocessed_data_folder"] = output_path

    #num_classes
    merge_DoDNetPlans_plans_3D["num_classes"] = len(merge_DoDNetPlans_plans_3D["dataset_properties"]["all_classes"])

    #all_classes
    merge_DoDNetPlans_plans_3D["all_classes"] = merge_DoDNetPlans_plans_3D["dataset_properties"]["all_classes"]

    #base_num_features
    merge_DoDNetPlans_plans_3D["base_num_features"] = 32

    #use_mask_for_norm
    merge_DoDNetPlans_plans_3D["use_mask_for_norm"] = OrderedDict([(0, False),(1, False),(2, False),(3, False)])

    #keep_only_largest_region
    merge_DoDNetPlans_plans_3D["keep_only_largest_region"] = None

    #min_region_size_per_class
    merge_DoDNetPlans_plans_3D["min_region_size_per_class"] = None

    #min_size_per_class
    merge_DoDNetPlans_plans_3D["min_size_per_class"] = None

    #transpose_forward
    merge_DoDNetPlans_plans_3D["transpose_forward"] = [0, 1, 2]

    #transpose_backward
    merge_DoDNetPlans_plans_3D["transpose_backward"] = [0, 1, 2]

    #data_identifier
    merge_DoDNetPlans_plans_3D["data_identifier"] = "DoDNetData_plans"

    #plans_per_stage
    merge_DoDNetPlans_plans_3D["plans_per_stage"] = {0: {'batch_size': 2, 'num_pool_per_axis': [4, 5, 5],
                                                    'patch_size': array([ 64, 192, 192]), 'median_patient_size_in_voxels':
                                                    array([ 29, 113, 133]), 'current_spacing': array([3. , 1.5, 1.5]),
                                                    'original_spacing': array([3. , 1.5, 1.5]), 'do_dummy_2D_data_aug':
                                                    False, 'pool_op_kernel_sizes': [[1, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
                                                    'conv_kernel_sizes': [[1, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]]}}

    #preprocessor_name
    merge_DoDNetPlans_plans_3D["preprocessor_name"] = "DoDNetPreprocessor"

    #conv_per_stage
    merge_DoDNetPlans_plans_3D["conv_per_stage"] = 2

    f = open(os.path.join(output_path, "DoDNetPlans_plans_3D.pkl"), "wb")
    write_pkl(merge_DoDNetPlans_plans_3D, f)

    f.close()

def merge_json_funtion(json_path_list, output_path):
    json_list = [json.load(open(os.path.join(json_path), 'r', encoding='utf8')) for json_path in json_path_list]
    merge_json = {}

    merge_json['name'] = "Merge Datasets"
    merge_json['description'] = "nothing"
    merge_json['tensorImageSize'] = "4D"
    merge_json['reference'] = ""
    merge_json['licence'] = ""
    merge_json['release'] = "0.0"
    merge_json['modality'] = {
        "0": ["T1", "T2", "CT"],
        "1": "T1ce",
        "2": "T2",
        "3": "FLAIR"
    }
    merge_json['labels'] = {i: str(i) for i in range(4)} #Maximum number of classes in all tasks
    merge_json['numTraining'] = 0
    merge_json['numTest'] = 0
    merge_json['training'] = []
    merge_json['test'] = []

    all_patient_names = []
    for i, sub_json in enumerate(json_list):
        merge_json['numTraining'] += sub_json['numTraining']
        merge_json['training'].extend(sub_json['training'])
        patient_names = [
            sample_dict["image"].split("/")[-1].split(".")[0]
            for sample_dict in sub_json['training']
        ]
        all_patient_names.append(patient_names)


    save_json(merge_json, join(output_path, "dataset.json"))

    return all_patient_names


def copy_data_and_collect_ids(data_path, out_path_label, out_path_image, ped=False):
    patient_names = []
    for sample in tqdm.tqdm(sorted(os.listdir(data_path))):
        if ".DS_Store" in sample:
            continue
        patient_names.append(sample)
        t1 = os.path.join(data_path, sample, "{}_t1.nii.gz".format(sample))
        t1c = os.path.join(data_path, sample, "{}_t1ce.nii.gz".format(sample))
        t2 = os.path.join(data_path, sample, "{}_t2.nii.gz".format(sample))
        flair = os.path.join(data_path, sample, "{}_flair.nii.gz".format(sample))
        seg = os.path.join(data_path, sample, "{}_seg.nii.gz".format(sample))

        shutil.copy(t1, os.path.join(out_path_image, sample + "_0000.nii.gz"))
        shutil.copy(t1c, os.path.join(out_path_image, sample + "_0001.nii.gz"))
        shutil.copy(t2, os.path.join(out_path_image, sample + "_0002.nii.gz"))
        shutil.copy(flair, os.path.join(out_path_image, sample + "_0003.nii.gz"))

        copy_BraTS_segmentation_and_convert_labels(seg, os.path.join(out_path_label, sample + ".nii.gz"), ped=ped)

    return patient_names

# patient_names = []
brats_patient_names = []
ped_patient_names = []

out_path = "/data/Task001_BraTS_OOD/"
# data_path = "./BraTS2021/MICCAI_BraTS2021_TrainingData/"
# brats_data_path = "/data/nnUNet_raw_data/Task021_BraTS2021"
brats_data_path = "/data/BraTS2021"
# ped_data_path = "/data/nnUNet_raw_data/Task022_PedBraTS2023"
ped_data_path = "/data/Task022_PedBraTS2023"

os.makedirs(out_path, exist_ok=True)
out_path_image = os.path.join(out_path, "imagesTr")
out_path_label = os.path.join(out_path, "labelsTr")

os.makedirs(out_path_image, exist_ok=True)
os.makedirs(out_path_label, exist_ok=True)
os.makedirs(os.path.join(out_path, "imagesTs"), exist_ok=True)


# brats_patient_names = copy_data_and_collect_ids(brats_data_path, out_path_label, out_path_image, ped=False)
# ped_patient_names = copy_data_and_collect_ids(ped_data_path, out_path_label, out_path_image, ped=True)
# all_patient_names = brats_patient_names + ped_patient_names 

# json_dict = OrderedDict()
# json_dict['name'] = "BraTS_OOD"
# json_dict['description'] = "nothing"
# json_dict['tensorImageSize'] = "4D"
# json_dict['reference'] = "see BraTS2021, BraTS2023, PedBraTS2023"
# json_dict['licence'] = "see Ped/BraTS2021/3 license"
# json_dict['release'] = "0.0"
# json_dict['modality'] = {
#     "0": "T1",
#     "1": "T1ce",
#     "2": "T2",
#     "3": "FLAIR"
# }
# json_dict['labels'] = {
#     # "-1": "anomaly", # maybe no need to include it as a "class"
#     "0": "background",
#     "1": "edema",
#     "2": "non-enhancing",
#     "3": "enhancing",
# }
# json_dict['numTraining'] = len(brats_patient_names)
# json_dict['numTest'] = 0
# json_dict['training'] = [{'image': "./imagesTr/%s.nii.gz" % i, "label": "./labelsTr/%s.nii.gz" % i} for i in
#                          all_patient_names]
# json_dict['test'] = []

# save_json(json_dict, join(out_path, "dataset.json"))




task_name  = "Task001_BraTS_OOD"
# if os.path.exists(task_name):
#     shutil.rmtree(task_name)
# os.makedirs(task_name, exist_ok=True)
# root_path = "/erwen_SSD/1T/nnUNet_preprocessed/"
root_path = "/data/" #os.environ['nnUNet_preprocessed']
out_path = os.path.join(root_path, task_name)
out_path_image = os.path.join(out_path, "DoDNetData_plans_stage0")
out_path_label = os.path.join(out_path, "gt_segmentations")

os.makedirs(out_path_image, exist_ok=True)
os.makedirs(out_path_label, exist_ok=True)

# sub_datasets = ["Task091_BraTS" "Task022_PedBraTS2023"] #
sub_datasets_full_path = ["/data/BraTS2021", "/data/Task022_PedBraTS2023"]

# dataset_properties_path_list = [os.path.join(root_path, sub_dataset, "dataset_properties.pkl") for sub_dataset in sub_datasets]
# DoDNetPlans_plans_3D_path_list = [os.path.join(root_path, sub_dataset, "DoDNetPlans_plans_3D.pkl") for sub_dataset in sub_datasets]
dataset_properties_path_list = [os.path.join(sub_dataset_full_path, "dataset_properties.pkl") for sub_dataset_full_path in sub_datasets_full_path]
DoDNetPlans_plans_3D_path_list = [os.path.join(sub_dataset_full_path, "DoDNetPlans_plans_3D.pkl") for sub_dataset_full_path in sub_datasets_full_path]
# dataset json files
# json_path_list = [os.path.join(root_path, sub_dataset, "dataset.json") for sub_dataset in sub_datasets]
json_path_list = [os.path.join(sub_dataset_full_path, "dataset.json") for sub_dataset_full_path in sub_datasets_full_path]
# splits files
# splits_final_path_list = [os.path.join(root_path, sub_dataset, "splits_final.pkl") for sub_dataset in sub_datasets]
# merge 
merge_dataset_properties_funtion(dataset_properties_path_list, output_path=out_path)
merge_DoDNetPlans_plans_3D_funtion(DoDNetPlans_plans_3D_path_list, output_path=out_path)
all_patient_names = merge_json_funtion(json_path_list, output_path=out_path)
brats_patient_names, ped_patient_names = tuple(all_patient_names)
# merge_splits_final_funtion(splits_final_path_list, output_path=out_path)

import random
random.seed(1234)
random.shuffle(brats_patient_names)
print("shuffle!")
data_split_train_val = OrderedDict()
with open(os.path.join(out_path, "splits_final.pkl"), "wb") as pk:
    data_split_train_val["train"] = brats_patient_names
    data_split_train_val["val"] = ped_patient_names
    pickle.dump([data_split_train_val], pk)



# for sub_dataset in sub_datasets:
    # source_image = os.path.join(root_path, sub_dataset, "DoDNetData_plans_stage0")
    # source_label = os.path.join(root_path, sub_dataset, "gt_segmentations")
for sub_dataset_full_path in sub_datasets_full_path:
    source_image = os.path.join(sub_dataset_full_path, "DoDNetData_plans_stage0")
    source_label = os.path.join(sub_dataset_full_path, "gt_segmentations")

    for file in tqdm(os.listdir(source_image)):
        shutil.copyfile(os.path.join(source_image, file), os.path.join(out_path_image, file))

    for file in tqdm(os.listdir(source_label)):
        shutil.copyfile(os.path.join(source_label, file), os.path.join(out_path_label, file))
