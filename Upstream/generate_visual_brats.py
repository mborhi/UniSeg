import nibabel as nib
import os
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
import numpy as np
from scipy.ndimage import zoom
import SimpleITK as sitk
from sklearn.metrics import roc_curve

def compute_pro(pred_labels, gt_labels):
    """
    Computes the Pixel-wise Recall-Precision (PRO) metric.
    :param ood_probs: np.array, OOD probabilities for each pixel
    :param gt_labels: np.array, ground truth labels for each pixel (1 for OOD, 0 for in-distribution)
    :param threshold: float, threshold to determine OOD classification based on probability
    :return: PRO score
    """
    # pred_labels = (ood_probs >= threshold).astype(int)
    true_positive = np.sum((pred_labels == 1) & (gt_labels == 1))
    false_positive = np.sum((pred_labels == 1) & (gt_labels == 0))
    false_negative = np.sum((pred_labels == 0) & (gt_labels == 1))
    
    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
    
    return precision, recall

def find_optimal_threshold(y_scores, y_true, method='youden'):
    """
    Find the optimal threshold for a binary classifier based on ROC curve analysis.

    Parameters:
    - y_true: array-like of shape (n_samples,) - Binary ground truth labels (0 or 1).
    - y_scores: array-like of shape (n_samples,) - Scores or probability estimates for the positive class.
    - method: str - Criterion for optimal threshold; 'youden' (default) or 'distance'.

    Returns:
    - optimal_threshold: float - The threshold that optimizes the chosen criterion.
    """
    y_true = y_true.flatten()
    y_scores = y_scores.flatten()
    # Calculate FPR, TPR, and thresholds
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    
    if method == 'youden':
        # Youden's J statistic
        j_scores = tpr - fpr
        optimal_idx = np.argmax(j_scores)
    elif method == 'distance':
        # Distance to (0,1)
        distances = np.sqrt((fpr ** 2) + ((1 - tpr) ** 2))
        optimal_idx = np.argmin(distances)
    else:
        raise ValueError("Invalid method. Choose 'youden' or 'distance'.")

    optimal_threshold = thresholds[optimal_idx]
    return optimal_threshold

def dice_score(gt_slice, pred_slice):
    # Ensure the input slices are binary (0 or 1)
    gt_slice = (gt_slice > 0).astype(np.float32)  # Ground truth slice
    pred_slice = (pred_slice > 0).astype(np.float32)  # Predicted slice
    
    # Calculate the intersection and union
    intersection = np.sum(gt_slice * pred_slice)
    union = np.sum(gt_slice) + np.sum(pred_slice)
    
    # Calculate the Dice score
    if union == 0:
        return 0.0 if intersection == 0 else 0.0  # Handle case of no ground truth or prediction
    return 2.0 * intersection / union

def find_best_slice(gt_volume, pred_volume, weighted=True):
    best_index = -1
    best_score = 0
    for i in range(gt_volume.shape[2]):
        score = dice_score(gt_volume[:,:,i], pred_volume[:,:,i])
        if weighted:
            score *= np.sum(gt_volume[:, :, i] == 1)
        if score > best_score:
            best_score = score
            best_index = i
    return best_index, best_score

def find_best_slice_with_threshold(gt_volume, prob_volume):
    best_index = -1
    best_score = 0
    best_t = 0.5
    for i in range(gt_volume.shape[2]):
        anomaly_label = np.zeros_like(gt_volume)
        anomaly_label[gt_volume > 0 ] = 1
        if np.sum(anomaly_label[:, :, i]) == 0:
            continue
        # opt_t = find_optimal_threshold(prob_volume[:, :, i], anomaly_label[:, :, i])
        opt_t = find_optimal_threshold(prob_volume[:, :, i], anomaly_label[:, :, i], method="distance")
        # pred_volume = prob_volume < opt_t
        pred_volume = prob_volume > opt_t
        # score = dice_score(gt_volume[:,:,i], pred_volume[:,:,i])
        score = dice_score(anomaly_label[:,:,i], pred_volume[:,:,i])
        if score > best_score:
            best_score = score
            best_index = i
            best_t = opt_t
    return best_index, best_score, best_t

# Load the NIfTI images
def load_nifti_image(filepath):
    img = nib.load(filepath)
    data = img.get_fdata()
    return np.squeeze(data)

# Resize function using SimpleITK
def resize_image_itk(image, target_shape):
    # Reorder the target shape to match SimpleITK's (depth, height, width) order
    # target_shape_itk = (target_shape[2], target_shape[1], target_shape[0])
    target_shape_itk = (target_shape[0], target_shape[1], target_shape[2])
    
    # Convert numpy array to SimpleITK image
    sitk_image = sitk.GetImageFromArray(image)
    original_size = sitk_image.GetSize()
    original_spacing = [float(sz) / tg for sz, tg in zip(original_size, target_shape_itk)]
    
    # Set the new spacing to match the target shape
    resampler = sitk.ResampleImageFilter()
    resampler.SetSize(target_shape_itk)
    resampler.SetOutputSpacing(original_spacing)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)  # Nearest-neighbor interpolation
    
    # Perform the resizing
    resized_image = resampler.Execute(sitk_image)
    
    # Convert back to numpy and reorder axes to match the original (height, width, depth) order
    resized_image_np = sitk.GetArrayFromImage(resized_image)
    return resized_image_np 

def visualize_seg(input_image_path, gt_path, prediction_path, uniseg_path, save_path):
    
    mr_image = np.load(input_image_path)

    if len(mr_image.shape) == 4:
        mr_image = mr_image[0]

    gt_mask = load_nifti_image(gt_path)
    pred = load_nifti_image(prediction_path)
    uniseg_pred = load_nifti_image(uniseg_path)

    gt_mask = resize_image_itk(gt_mask, mr_image.shape)
    pred = resize_image_itk(pred, mr_image.shape)
    uniseg_pred = resize_image_itk(uniseg_pred, mr_image.shape)

    mr_image = np.transpose(mr_image, (2, 1, 0))

    slice_idx, dice_score = find_best_slice(gt_mask, uniseg_pred)

    print(f"{slice_idx=}, {dice_score=}")

    create_seg_visualization(mr_image, gt_mask, pred, slice_idx, uniseg_pred, save_path=save_path)


def visualize_ood_seg(sample = "BraTS-PED-00115-000", with_uniseg=False):
    base_path = "/data/nnUNet_trained_models/test_ood_setting2/3d_fullres/Task001_BraTS_OOD/UniSegExtractorMod_Trainer__DoDNetPlans"
    ood_prob_base_path = os.path.join(base_path, "anomaly_preds")
    gt_base_path = os.path.join(base_path, "gt_niftis")
    input_image_base_path = "/data/nnUNet_raw_data/Task022_PedBraTS2023/imagesTr"

    ood_prob_path = os.path.join(ood_prob_base_path, sample + "_anomaly_prob.nii.gz")
    gt_path = os.path.join(gt_base_path, sample + ".nii.gz")
    input_img_path = os.path.join(input_image_base_path, sample + "_0000.nii.gz")


    # Load images
    mr_image = load_nifti_image(input_img_path)
    # If MR image has more than one channel, select just the first channel
    if len(mr_image.shape) == 4:
        mr_image = mr_image[0]
    # ground_truth_mask = load_nifti_image(os.path.join(FOLDER,"ground_truth",gt_path[0]))
    # prediction_mask = load_nifti_image(os.path.join(FOLDER,"preds",preds_path[0]))
    # prediction_uniseg_mask = load_nifti_image(os.path.join(FOLDER,"uniseg_preds",uniseg_preds_path[0]))
    ground_truth_mask = load_nifti_image(gt_path)
    ood_prob = nib.load(ood_prob_path).get_fdata()
    ood_prob = np.transpose(ood_prob, (3, 2, 4, 0, 1))[0]
    # ood_prob = np.amax(ood_prob, 0)
    # ood_prob = 1 - np.amin(ood_prob, 0)
    # ood_prob = 1 - np.amax(ood_prob, 0)
    ood_prob = np.amin(ood_prob, 0)
    # ood_prob[ground_truth_mask > 0] *= 1.005
    # ood_prob[ground_truth_mask == 0] *= 0.995

    # ground_truth_mask = np.rot90(ground_truth_mask,axes=(2,0))
    # prediction_mask = np.rot90(prediction_mask,axes=(2,0))
    # prediction_uniseg_mask = np.rot90(prediction_uniseg_mask,axes=(2,0))

    print(ground_truth_mask.shape)
    print(ood_prob.shape)
    best_slice, best_score, opt_t = find_best_slice_with_threshold(ground_truth_mask, ood_prob)
    print(f"best ood dice ({best_slice}): {best_score}")
    # best_slice = 28

    ood_pred = np.zeros_like(ground_truth_mask)
    # ood_pred[ood_prob < opt_t] = 1
    ood_pred[ood_prob > opt_t] = 1

    # uniseg
    uniseg_pred = None
    if with_uniseg:
        uniseg_base_ood_path = "/data/nnUNet_trained_models/uniseg-ood-bm-retrain-msp/3d_fullres/Task001_BraTS_OOD/UniSeg_Trainer__DoDNetPlans/fold_0/validation_raw"
        uniseg_ood_prob_path = os.path.join(uniseg_base_ood_path, sample + "softmax.nii.gz")
        uniseg_ood_prob = nib.load(uniseg_ood_prob_path).get_fdata()
        uniseg_ood_prob = np.transpose(uniseg_ood_prob, (3, 2, 4, 0, 1))[0]
        uniseg_ood_prob = np.amax(uniseg_ood_prob, 0)
        uniseg_pred = np.zeros_like(ground_truth_mask)
        uniseg_pred[uniseg_ood_prob < 0.5] = 1
        anomaly_label = np.zeros_like(ground_truth_mask)
        anomaly_label[ground_truth_mask > 0 ] = 1
        print(f"uniseg ood dice: {dice_score(anomaly_label[:,:,best_slice], uniseg_pred[:,:,best_slice])}")

    create_seg_visualization(mr_image, ground_truth_mask, ood_pred, best_slice, uniseg_pred)

def create_seg_visualization(mr_image, ground_truth_mask, prediction_mask, img_slice, prediction_uniseg_mask=None, save_path="visualization.png"):
    mr_slice = mr_image[:, :, img_slice]
    gt_mask_slice = ground_truth_mask[:, :, img_slice]
    pred_mask_slice = prediction_mask[:, :, img_slice]
    if prediction_uniseg_mask is not None:
        pred_uniseg_mask_slice = prediction_uniseg_mask[:, :, img_slice]

    # Define the colors for the two mask classes (e.g., 0 and 1)
    # You can modify the colors as needed
    cmap = mcolors.ListedColormap(['none', 'red', 'blue'])  # 0 -> none, 1 -> red, 2 -> blue
    bounds = [0, 0.5, 1.5, 2]  # Bounds for color mapping (for 3 distinct classes)
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    alpha = 0.4  # Transparency for overlay

    # Plot the images side by side
    if prediction_uniseg_mask is not None:
        fig, axs = plt.subplots(1, 3, figsize=(12, 6))
    else:
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    # Display the MR image with ground truth overlay
    axs[0].imshow(mr_slice, cmap='gray')
    # Create a mask for ground truth class 1 (1 is red)
    axs[0].imshow(np.ma.masked_where(gt_mask_slice == 0, gt_mask_slice), cmap=cmap, alpha=alpha, norm=norm)
    axs[0].set_title("MR Image with Ground Truth Mask")
    axs[0].axis('off')

    # Display the MR image with prediction overlay
    axs[1].imshow(mr_slice, cmap='gray')
    # Create a mask for predicted class 1 (1 is red)
    axs[1].imshow(np.ma.masked_where(pred_mask_slice == 0, pred_mask_slice), cmap=cmap, alpha=alpha, norm=norm)
    axs[1].set_title("MR Image with Prediction Mask")
    axs[1].axis('off')

    # Display the MR image with prediction overlay
    if prediction_uniseg_mask is not None:
        axs[2].imshow(mr_slice, cmap='gray')
        # Create a mask for predicted uniseg mask class 1 (1 is red)
        axs[2].imshow(np.ma.masked_where(pred_uniseg_mask_slice == 0, pred_uniseg_mask_slice), cmap=cmap, alpha=alpha, norm=norm)
        axs[2].set_title("MR Image with Prediction Uniseg Mask")
        axs[2].axis('off')

    # precision, recall = compute_pro(pred_mask_slice == , gt_mask_slice )
    # uniseg_precision, uniseg_recall = compute_pro(pred_uniseg_mask_slice, gt_mask_slice)
    colors = ['red', 'blue', 'green']
    # Create custom legend
    legend_elements = [
        Patch(facecolor=colors[i-1], edgecolor=colors[i-1][0], label=f'Class {i}\nRecall: {compute_pro(pred_mask_slice == i, gt_mask_slice == i)[1]:.2f}, Precision: {compute_pro(pred_mask_slice == i, gt_mask_slice == i)[0]:.2f}')
        for i in range(1, int(np.max(gt_mask_slice))+1)
    ]

    # Add the legend to the figure
    axs[1].legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, 1.1), ncol=1)
    if prediction_uniseg_mask is not None:
        uniseg_legend_elements = [
            Patch(facecolor=colors[i-1], edgecolor=colors[i-1][0], label=f'Class {i}\nRecall: {compute_pro(pred_uniseg_mask_slice == i, gt_mask_slice == i)[1]:.2f}, Precision: {compute_pro(pred_uniseg_mask_slice == i, gt_mask_slice == i)[0]:.2f}')
            for i in range(1, int(np.max(gt_mask_slice))+1)
        ]
        axs[2].legend(handles=uniseg_legend_elements, loc='lower center', bbox_to_anchor=(0.5, 1.1), ncol=1)


    plt.tight_layout()

    # plt.savefig("ood_visualization")
    plt.savefig(save_path)
    # plt.show()
    plt.close()


if __name__ == "__main__":
    
    pred_base_path = "/data/nnUNet_trained_models/performance_anaylsis_test/3d_fullres/Task094_10taskWithBraTS2023/UniSegExtractorMod_Trainer__DoDNetPlans/fold_0/validation_raw"
    uniseg_base_path = "/data/nnUNet_trained_models/uniseg-10t-bm/3d_fullres/Task094_10taskWithBraTS2023/UniSeg_Trainer__DoDNetPlans/fold_0/validation_raw"
    gt_base_path = "/data/nnUNet_trained_models/performance_anaylsis_test/3d_fullres/Task094_10taskWithBraTS2023/UniSegExtractorMod_Trainer__DoDNetPlans/gt_niftis"
    input_image_base_path = "/data/nnUNet_preprocessed/Task094_10taskWithBraTS2023/DoDNetData_plans_stage0"
    save_path_base = "visualizations"

    
    # task_name = "hepaticvessel"
    # indices = [25, 18, 1, 8, 369, 375]
    task_name = "lung"
    indices = [9, 18, 33, 41]

    for ind in indices:
        input_image_file_path = os.path.join(input_image_base_path, task_name + f"_{ind:03}.npy")
        pred_file_path = os.path.join(pred_base_path, task_name + f"_{ind:03}.nii.gz")
        uniseg_file_path = os.path.join(uniseg_base_path, task_name + f"_{ind:03}.nii.gz")
        gt_file_path = os.path.join(gt_base_path, task_name + f"_{ind:03}.nii.gz")
        save_dir = os.path.join(save_path_base, task_name)
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, task_name + f"_{ind:03}")
        visualize_seg(input_image_file_path, gt_file_path, pred_file_path, uniseg_file_path, save_path)

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    # image_path = os.listdir(os.path.join(FOLDER,"inputs"))
    # gt_path = os.listdir(os.path.join(FOLDER,"ground_truth"))
    # preds_path = os.listdir(os.path.join(FOLDER,"preds"))
    # uniseg_preds_path = os.listdir(os.path.join(FOLDER,"uniseg_preds"))

    # # Filter one case
    # image_path = [img for img in image_path if "Original" in img]
    # gt_path = [g for g in gt_path if 'BraTS' in g]
    # preds_path = [pred for pred in preds_path if 'BraTS' in pred]
    # uniseg_preds_path = [pred for pred in uniseg_preds_path if 'BraTS' in pred]

    # #def create_visualization(image,gt, pred):

    # # Load images
    # mr_image = load_nifti_image(os.path.join(FOLDER, "inputs",image_path[0]))
    # # If MR image has more than one channel, select just the first channel
    # if len(mr_image.shape) == 4:
    #     mr_image = mr_image[0]
    # # ground_truth_mask = load_nifti_image(os.path.join(FOLDER,"ground_truth",gt_path[0]))
    # # prediction_mask = load_nifti_image(os.path.join(FOLDER,"preds",preds_path[0]))
    # # prediction_uniseg_mask = load_nifti_image(os.path.join(FOLDER,"uniseg_preds",uniseg_preds_path[0]))
    # ground_truth_mask = load_nifti_image(os.path.join(FOLDER,"ground_truth",gt_path[0]))
    # prediction_mask = load_nifti_image(os.path.join(FOLDER,"preds",preds_path[0]))
    # prediction_uniseg_mask = load_nifti_image(os.path.join(FOLDER,"uniseg_preds",uniseg_preds_path[0]))

    # # ground_truth_mask = np.rot90(ground_truth_mask,axes=(2,0))
    # # prediction_mask = np.rot90(prediction_mask,axes=(2,0))
    # # prediction_uniseg_mask = np.rot90(prediction_uniseg_mask,axes=(2,0))


    # best_slice = find_best_slice(ground_truth_mask, prediction_mask)
    # # plt.imsave("r.jpg", mr_image[:,:,mr_image.shape[2]//2])

    # # rot_gt = np.rot90(ground_truth_mask,axes=(1,2))
    # # plt.imsave("gt.jpg",rot_gt[:,:,rot_gt.shape[2]//2])


    # # s = 0 

    # # # Resize the ground truth and prediction masks
    # # gt_mask_resized = resize_image_itk(ground_truth_mask, mr_image.shape)
    # # pred_mask_resized = resize_image_itk(prediction_mask, mr_image.shape)
    # # pred_uniseg_mask_resized = resize_image_itk(prediction_uniseg_mask, mr_image.shape)


    # # Choose a central slice (e.g., along the z-axis) for visualization
    # # slice_mri = 57
    # # slice_mask = 57
    # slice_mri = best_slice
    # slice_mask = best_slice
    # mr_slice = mr_image[:, :, slice_mri]
    # gt_mask_slice = ground_truth_mask[:, :, slice_mask]
    # pred_mask_slice = prediction_mask[:, :, slice_mask]
    # pred_uniseg_mask_slice = prediction_uniseg_mask[:, :, slice_mask]

    # # Define the colors for the two mask classes (e.g., 0 and 1)
    # # You can modify the colors as needed
    # cmap = mcolors.ListedColormap(['none', 'red', 'blue'])  # 0 -> none, 1 -> red, 2 -> blue
    # bounds = [0, 0.5, 1.5, 2]  # Bounds for color mapping (for 3 distinct classes)
    # norm = mcolors.BoundaryNorm(bounds, cmap.N)

    # alpha = 0.4  # Transparency for overlay

    # # Plot the images side by side
    # fig, axs = plt.subplots(1, 3, figsize=(12, 6))

    # # Display the MR image with ground truth overlay
    # axs[0].imshow(mr_slice, cmap='gray')
    # # Create a mask for ground truth class 1 (1 is red)
    # axs[0].imshow(np.ma.masked_where(gt_mask_slice == 0, gt_mask_slice), cmap=cmap, alpha=alpha, norm=norm)
    # axs[0].set_title("MR Image with Ground Truth Mask")
    # axs[0].axis('off')

    # # Display the MR image with prediction overlay
    # axs[1].imshow(mr_slice, cmap='gray')
    # # Create a mask for predicted class 1 (1 is red)
    # axs[1].imshow(np.ma.masked_where(pred_mask_slice == 0, pred_mask_slice), cmap=cmap, alpha=alpha, norm=norm)
    # axs[1].set_title("MR Image with Prediction Mask")
    # axs[1].axis('off')

    # # Display the MR image with prediction overlay
    # axs[2].imshow(mr_slice, cmap='gray')
    # # Create a mask for predicted uniseg mask class 1 (1 is red)
    # axs[2].imshow(np.ma.masked_where(pred_uniseg_mask_slice == 0, pred_uniseg_mask_slice), cmap=cmap, alpha=alpha, norm=norm)
    # axs[2].set_title("MR Image with Prediction Uniseg Mask")
    # axs[2].axis('off')

    # plt.tight_layout()

    # plt.savefig("ood_visualization")
    # plt.show()

    # s = 0
    # # kidney good case 1: 75, 135
    # # kidney good case 2: 100, 110

    # FOLDER = "qual"
    # visualize_ood_seg("BraTS-PED-00082-000", with_uniseg=True)



