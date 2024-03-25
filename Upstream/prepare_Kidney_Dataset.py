import os
import shutil

# Note that the output paths of the preprocessed datasets should be in the $nnUNet_raw_data_base/nnUNet_raw_data/ directory.
base_path = os.environ['nnUNet_raw_data_base']
output = os.path.join(os.environ['nnUNet_preprocessed'], os.environ['nnUNet_raw_data_base'])
path = os.path.join(base_path, "MOTS/Kidney/kits19/data")
# path = "/media/userdisk1/yeyiwen/nnUNetFrame/DATASET/nnUNet_raw_data_base/nnUNet_raw_data/1Kidney/origin/"
# output = "/media/userdisk1/yeyiwen/nnUNetFrame/DATASET/nnUNet_raw_data_base/nnUNet_raw_data/1Kidney/"
imagesTr = os.path.join(output, "imagesTr")
labelsTr = os.path.join(output, "labelsTr")
if not os.path.exists(imagesTr):
    os.makedirs(imagesTr)
if not os.path.exists(labelsTr):
    os.makedirs(labelsTr)
for case in os.listdir(path):
    patient_path = os.path.join(path, case)
    if not os.path.isfile(os.path.join(patient_path, "segmentation.nii.gz")):
        continue
    shutil.copyfile(os.path.join(patient_path, "imaging.nii.gz"), os.path.join(imagesTr, "kidney_{}.nii.gz".format(int(case.split("_")[-1]))))
    shutil.copyfile(os.path.join(patient_path, "segmentation.nii.gz"),
                    os.path.join(labelsTr, "kidney_{}.nii.gz".format(int(case.split("_")[-1]))))

