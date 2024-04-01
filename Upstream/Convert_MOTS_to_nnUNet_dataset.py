import os
import shutil
from batchgenerators.utilities.file_and_folder_operations import *
from collections import OrderedDict
from tqdm import tqdm

# "/data/uniseg/MOTS_formatted" => -v /media/.../uniseg:/data
print(os.listdir(os.environ['nnUNet_raw_data_base']))
data_path = os.path.join(os.environ['nnUNet_raw_data_base'], 'MOTS_formatted')
# data_path = "/media/userdisk1/yeyiwen/nnUNetFrame/DATASET/nnUNet_raw/nnUNet_raw_data/"
# sub_dataset_name = ["Task32_HepaticVessel", "Task33_Pancreas", "Task34_Colon", "Task35_Lung", "Task36_Spleen"]
sub_dataset_name = ["Task30_Liver", "Task31_Kidney", "Task32_HepaticVessel", "Task33_Pancreas", "Task34_Colon", "Task35_Lung", "Task36_Spleen"]
# sub_dataset_name = ["Task30_Liver", "Task31_Kidney"]
# out_path = os.path.join(os.environ['nnUNet_raw_data_base'], "uniform_uniseg_clip", "")
out_path = os.path.join(os.environ['nnUNet_raw_data_base'], "nnUNet_raw_data", "Task091_MOTS")
if not os.path.exists(out_path):
    # shutil.rmtree(out_path)
    os.makedirs(out_path)
if not os.path.exists(out_path):
    os.makedirs(out_path)
out_image_path = os.path.join(out_path, "imagesTr")
out_label_path = os.path.join(out_path, "labelsTr")
if not os.path.exists(out_image_path):
    os.makedirs(out_image_path)
if not os.path.exists(out_label_path):
    os.makedirs(out_label_path)
if not os.path.exists(os.path.join(out_path, "imagesTs")):
    os.makedirs(os.path.join(out_path, "imagesTs"))

patient_names = []
for dataset in sub_dataset_name:
    sub_dataset_image_path = os.path.join(data_path, dataset, "imagesTr")
    sub_dataset_label_path = os.path.join(data_path, dataset, "labelsTr")
    for name in tqdm(os.listdir(sub_dataset_image_path)):
        if ".nii.gz" not in name:
            continue
        pname = name.replace(".nii", "").replace(".gz", "")
        if dataset == "Task30_Liver" and "volume" in pname:
            pname = pname.replace("volume", "liver").replace("-", "_")
            label_name = name.replace("volume", "segmentation")
        else :
            label_name = name
        if os.path.isfile(os.path.join(out_label_path, pname+".nii.gz")):
            patient_names.append(pname)
            continue
        # elif (name.endswith(".nii.gz") or name.endswith(".nii")) and not os.path.isfile(os.path.join(out_label_path, name)):
        print(name)
        print(pname)
        shutil.copy(os.path.join(sub_dataset_image_path, name), os.path.join(out_image_path, pname + "_0000.nii.gz"))
        shutil.copy(os.path.join(sub_dataset_label_path, label_name), os.path.join(out_label_path, pname+".nii.gz"))
        # patient_names.append(name.replace(".nii.gz", ""))
        patient_names.append(pname)


json_dict = OrderedDict()
json_dict['name'] = "MOTS benchmark for multi-organ and tumor segmentation"
json_dict['description'] = "nothing"
json_dict['tensorImageSize'] = "4D"
json_dict['reference'] = "see https://github.com/jianpengz/DoDNet"
json_dict['licence'] = "see https://github.com/jianpengz/DoDNet"
json_dict['release'] = "0.0"
json_dict['modality'] = {
    "0": "CT"
}
json_dict['labels'] = {
    "0": "background",
    "1": "organ",
    "2": "tumor"
}
json_dict['numTraining'] = len(patient_names)
json_dict['numTest'] = 0
json_dict['training'] = [{'image': "./imagesTr/%s.nii.gz" % i, "label": "./labelsTr/%s.nii.gz" % i} for i in
                         patient_names]
json_dict['test'] = []

save_json(json_dict, join(out_path, "dataset.json"))


#we provide data split following DoDNet