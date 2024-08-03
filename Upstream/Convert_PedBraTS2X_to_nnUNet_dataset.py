import os
import shutil
import tqdm
import numpy as np
import SimpleITK as sitk
from PIL import Image
from collections import OrderedDict
from batchgenerators.utilities.file_and_folder_operations import *

def copy_BraTS_segmentation_and_convert_labels(in_file, out_file):
    # use this for segmentation only!!!
    # nnUNet wants the labels to be continuous. BraTS is 0, 1, 2, 4 -> we make that into 0, 1, 2, 3
    img = sitk.ReadImage(in_file)
    img_npy = sitk.GetArrayFromImage(img)

    img_corr = sitk.GetImageFromArray(img_npy)
    uniques = np.unique(img_npy)
    # check if it needs to be relabeld
    # for u in uniques:
    #     if u not in [0, 1, 2, 4]:
    #         raise RuntimeError('unexpected label')

    #     seg_new = np.zeros_like(img_npy)
    #     seg_new[img_npy == 4] = 3
    #     seg_new[img_npy == 2] = 1
    #     seg_new[img_npy == 1] = 2
    #     img_corr = sitk.GetImageFromArray(seg_new)
    
    img_corr.CopyInformation(img)
    sitk.WriteImage(img_corr, out_file)

patient_names = []

out_path = "/data/nnUNet_raw_data/Task022_PedBraTS2023/"
data_path = "/data_ped_brats/ped-brats2023/all_samples/"

os.makedirs(out_path, exist_ok=True)
out_path_image = os.path.join(out_path, "imagesTr")
out_path_label = os.path.join(out_path, "labelsTr")

os.makedirs(out_path_image, exist_ok=True)
os.makedirs(out_path_label, exist_ok=True)
os.makedirs(os.path.join(out_path, "imagesTs"), exist_ok=True)


for sample in tqdm.tqdm(sorted(os.listdir(data_path))):
    if ".DS_Store" in sample:
        continue
    
    seg = os.path.join(data_path, sample, "{}-seg.nii.gz".format(sample))
    
    if not os.path.isfile(seg):
        continue
    
    patient_names.append(sample)
    t1 = os.path.join(data_path, sample, "{}-t1n.nii.gz".format(sample))
    t1c = os.path.join(data_path, sample, "{}-t1c.nii.gz".format(sample))
    t2 = os.path.join(data_path, sample, "{}-t2w.nii.gz".format(sample))
    flair = os.path.join(data_path, sample, "{}-t2f.nii.gz".format(sample))

    shutil.copy(t1, os.path.join(out_path_image, sample + "_0000.nii.gz"))
    shutil.copy(t1c, os.path.join(out_path_image, sample + "_0001.nii.gz"))
    shutil.copy(t2, os.path.join(out_path_image, sample + "_0002.nii.gz"))
    shutil.copy(flair, os.path.join(out_path_image, sample + "_0003.nii.gz"))

    copy_BraTS_segmentation_and_convert_labels(seg, os.path.join(out_path_label, sample + ".nii.gz"))

json_dict = OrderedDict()
json_dict['name'] = "PedBraTS2023"
json_dict['description'] = "nothing"
json_dict['tensorImageSize'] = "4D"
json_dict['reference'] = "see Ped BraTS2023"
json_dict['licence'] = "see PedBraTS2024 license"
json_dict['release'] = "0.0"
json_dict['modality'] = {
    "0": "T1",
    "1": "T1ce",
    "2": "T2",
    "3": "FLAIR"
}
json_dict['labels'] = {
    "0": "background",
    "1": "edema",
    "2": "non-enhancing",
    "3": "enhancing",
}
json_dict['numTraining'] = len(patient_names)
json_dict['numTest'] = 0
json_dict['training'] = [{'image': "./imagesTr/%s.nii.gz" % i, "label": "./labelsTr/%s.nii.gz" % i} for i in
                         patient_names]
json_dict['test'] = []

save_json(json_dict, join(out_path, "dataset.json"))

import random
random.seed(1234)
random.shuffle(patient_names)
print("shuffle!")
data_split_train_val = OrderedDict()
with open(os.path.join(out_path, "splits_final.pkl"), "wb") as pk:
    data_split_train_val["train"] = patient_names[:int(0.8*len(patient_names))]
    data_split_train_val["val"] = patient_names[int(0.8*len(patient_names)):]
    pickle.dump([data_split_train_val], pk)
