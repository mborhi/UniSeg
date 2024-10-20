import wandb 
import numpy as np
import torch
from PIL import Image as PILImage


from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

# NOTE : gt, mask, seg_prediction, prob_predictions, feature_predictions 
def wandb_log(image, mask, pred):
    mask_ = mask[0, :, :].detach().cpu().numpy()
    # pred_ = torch.where(torch.sigmoid(pred[0, :, :])>=0.5, 1, 0).detach().cpu().numpy().astype(float) 
    pred_ = torch.where(torch.sigmoid(pred[0, :, :])>=0.5, 1, 0).detach().cpu().numpy().astype(float) 
    # NOTE we have this through log-likelihood
    pred_logit = torch.sigmoid(pred[0, :, :]).detach().cpu().numpy().astype(float)*255 
    final_mask_wandb = mask_*255
    final_pred_wandb = pred_*255
    wandb.log({"train": [wandb.Image(image, caption="image"),
                            wandb.Image(image, masks={"ground_truth": {"mask_data": final_mask_wandb}}, caption="label"),
                            wandb.Image(pred_logit, caption="logit"),
                            wandb.Image(image, masks={"predictions": {"mask_data": final_pred_wandb}}, caption="prediction")
            ]})


def wb_mask(bg_img, mask):
    return wandb.Image(bg_img, masks={
    "ground truth" : {"mask_data" : mask, "class_labels" : {0: "background", 1: "mask"} }})

def wandb_log_outputs(input_image, output_features, output_segmentation, output_probabilities, ground_truth_segmentation):
    num_classes = len(ground_truth_segmentation.unique().detach().cpu().numpy())
    output_segmentation = output_segmentation.long()
    # output_features = output_features.permute()
    input_image = input_image.detach().cpu().numpy()
    # output_features = output_features.detach().cpu()#.numpy()
    print(output_segmentation.shape)
    output_segmentation = output_segmentation.detach().cpu().numpy()
    # output_probabilities = output_segmentation#output_probabilities.detach().cpu().numpy()
    ground_truth_segmentation = ground_truth_segmentation.detach().cpu().numpy()

    gt_seg_wandb = ground_truth_segmentation * 255
    final_pred_wandb = output_segmentation * 255

    # compressed_output_features = pca_compress(output_features).cpu().numpy() # Note: output_features must have shape (B, H, W, feature_space_dim)
    wandb_img_logs = []
    wandb_mask_logs = []
    for slice in range(input_image.shape[0]):
        img = input_image[:, slice, :, :]
        gt_seg = ground_truth_segmentation[:, slice, :, :]

        wandb_img_logs.append(wandb.Image(
            img, caption=f"Slice: {slice}"
        ))
        wandb_mask_logs.append(wb_mask(img, gt_seg))
    
        # wandb.log({"train": [wandb.Image(input_image, caption="input_image"),
        #                         wandb.Image(input_image, masks={"ground_truth": {"mask_data": gt_seg_wandb}}, caption="label"),
        #                         # wandb.Image(output_probabilities, caption="class_wise_probabilities"),
        #                         # wandb.Image(compressed_output_features, caption="output_features"),
        #                         wandb.Image(input_image, masks={"predictions": {"mask_data": final_pred_wandb}}, caption="prediction")
        # ]})
    wandb.log({"Image": wandb_img_logs})
    wandb.log({"Segmentation mask": wandb_mask_logs})


def test_img_log_color(imgs, gt_segs, probs, tsk_idx, classes):
    class_labels = {1: "organ", 2: "tumor", 3: "brats", 4:"brats_1"}
    # num_classes = len(classes)
    # print(f"gt_segs: {gt_segs.shape} | {imgs.shape}")
    for batch_idx in range(imgs.shape[0]):
        # for cls in classes:
        for cls in range(1, classes):
            # gt_seg = gt_segs.argmax(1)[batch_idx].detach().cpu().numpy()
            # print(f"{gt_segs.shape}, {batch_idx}")
            # print(f"{gt_segs[batch_idx].shape}")
            img = imgs[batch_idx].amax(0)
            gt_seg = gt_segs[batch_idx].clone().amax(0).detach().cpu().numpy()
            slice_index, max_class_slice, max_voxel_count = find_slice_with_most_class_voxels(gt_seg, cls=cls)

            # print(f"Slice {slice_index} has the most voxels with class == {cls} ({max_voxel_count} voxels).")

            # image = rescale(img[batch_idx, :, slice_index, :].detach().cpu().numpy())#/255
            image = rescale(img[slice_index, :, :].detach().cpu().numpy())#/255
            # print(f"image: {image.shape}")
            
            # final_pred_wandb = probs.argmax(1)[batch_idx, :, slice_index, :].detach().cpu().numpy().astype(float)
            final_pred_wandb = probs.argmax(1)[batch_idx, slice_index, :, :].detach().cpu().numpy().astype(float)
            final_pred_wandb = rescale(final_pred_wandb)/255
            # print(f"final_pred_wandb: {final_pred_wandb.shape}")

            # Apply the color map to ground truth mask (RGB conversion)
            # final_mask_wandb = apply_color_map(gt_seg[:, slice_index, :], num_classes)
            # final_mask_wandb = gt_seg[:, slice_index, :]
            # final_mask_wandb = rescale(gt_seg[slice_index, :, :].astype(float))/255
            # final_mask_wandb = gt_segs.argmax(1)[batch_idx, slice_index, :, :].detach().cpu().numpy().astype(float)
            # final_mask_wandb = rescale(final_mask_wandb)/255
            final_mask_wandb = rescale(max_class_slice)/255 #rescale(final_mask_wandb)/255
            # final_mask_wandb = rescale(gt_seg[slice_index, :, :])/255

            # Apply the color map to the prediction mask (RGB conversion)
            # pred_mask_rgb = apply_color_map(final_pred_wandb, num_classes)
            # print(f"pred_mask_rgb: {pred_mask_rgb.shape}")

            # pred_logit = probs.amax(1)[batch_idx, :, slice_index, :].detach().cpu().numpy().astype(float)
            pred_logit = probs.amax(1)[batch_idx, slice_index, :, :].detach().cpu().numpy().astype(float)
            pred_logit = rescale(pred_logit)
            
            # Log images and masks as RGB
            wandb.log({f"train_{tsk_idx}": [wandb.Image(image, caption="image"),
                                    wandb.Image(image, masks={"ground_truth": {"mask_data": final_mask_wandb, "class_labels": class_labels}}, caption="ground truth"),
                                    wandb.Image(pred_logit, caption="logit"),
                                    wandb.Image(image, masks={"predictions": {"mask_data": final_pred_wandb, "class_labels": class_labels}}, caption="prediction")
                    ]})
# Define a simple color map (class index to RGB value)
# Adjust the color map as needed for your specific classes.
def apply_color_map(mask, num_classes):
    color_map = np.array([
        [0, 0, 0],       # Background (class 0) - black
        [255, 0, 0],     # Class 1 - red
        [0, 255, 0],     # Class 2 - green
        [0, 0, 255],     # Class 3 - blue
        # Add more colors if you have more classes
    ], dtype=np.uint8)

    rgb_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)  # Create an empty RGB mask
    for i in range(num_classes):
        rgb_mask[mask == i] = color_map[i]  # Assign color based on class index

    return rgb_mask


def test_img_log(img, gt_segs, probs, tsk_idx, classes):
    print(f"img: {img.shape}")
    print(f"gt_seg: {gt_segs.shape}")
    print(f"probs: {probs.shape}")
    # image = np.random.randint(low=0, high=256, size=(100, 100, 3))

    for batch_idx in range(img.shape[0]):
        for cls in classes:
            gt_seg = gt_segs.amax(1)[batch_idx].detach().cpu().numpy()
            # Find the slice with the most voxels where class > 0
            slice_index, max_class_slice, max_voxel_count = find_slice_with_most_class_voxels(gt_seg, cls=cls)

            print(f"Slice {slice_index} has the most voxels with class == 1 ({max_voxel_count} voxels).")

            # image = img[0, :, 100, :].detach().cpu().numpy()#np.random.randint(low=0, high=256, size=(100, 100, 3))
            image = rescale(img[batch_idx, :, slice_index, :].detach().cpu().numpy())#np.random.randint(low=0, high=256, size=(100, 100, 3))
            print(f"image: {image.shape}")
            # final_pred_wandb = probs.argmax(1)[0, :, 100, :].detach().cpu().numpy().astype(float)*255  #np.random.randint(low=0, high=255, size=(100, 100))
            final_pred_wandb = probs.argmax(1)[batch_idx, :, slice_index, :].detach().cpu().numpy().astype(float)#*255  #np.random.randint(low=0, high=255, size=(100, 100))
            final_pred_wandb = rescale(final_pred_wandb)
            # final_mask_wandb = np.random.choice([0, 1], size=(100, 100)) * 255
            # final_mask_wandb = gt_seg.argmax(1)[0, :, 100, :].detach().cpu().numpy() * 255
            final_mask_wandb = gt_seg[:, slice_index, :]# * 255
            final_mask_wandb = rescale(final_mask_wandb)
            # pred_logit = probs.amax(1)[0, :, 100, :].detach().cpu().numpy().astype(float)*255 
            pred_logit = probs.amax(1)[batch_idx, :, slice_index, :].detach().cpu().numpy().astype(float)#*255 
            pred_logit = rescale(pred_logit)
            wandb.log({f"train_{tsk_idx}": [wandb.Image(image, caption="image"),
                                    wandb.Image(image, masks={"ground_truth": {"mask_data": final_mask_wandb}}, caption="label"),
                                    wandb.Image(pred_logit, caption="logit"),
                                    wandb.Image(image, masks={"predictions": {"mask_data": final_pred_wandb}}, caption="prediction")
                    ]})

def find_slice_with_most_class_voxels(volume, cls=0):
    """
    Find the 2D slice along the first axis (axis=0) with the most voxels of a given class.
    
    Parameters:
    - volume: 3D numpy array (H, W, D) where H is along axis 0.
    - cls: The class to find the slice for (default is 0).
    
    Returns:
    - max_voxel_slice_index: The index of the slice along axis 0 with the most class voxels.
    - max_class_slice: The 2D slice with the maximum number of voxels of the given class.
    - max_voxel_count: The number of voxels of the class in that slice.
    """
    print(f"uniques: {np.unique(volume)}")
    # Count the number of voxels equal to `cls` in each slice along axis 0 (1st axis)
    voxel_counts = np.sum(volume == cls, axis=(1, 2))  # Summing along axes 1 and 2 gives the count for each 2D slice
    
    # Find the index of the slice with the maximum count of voxels of class `cls`
    max_voxel_slice_index = np.argmax(voxel_counts)
    
    # Retrieve the slice with the maximum number of voxels of class `cls`
    max_class_slice = volume[max_voxel_slice_index]
    
    # Get the count of class voxels in that slice
    max_voxel_count = voxel_counts[max_voxel_slice_index]
    
    return max_voxel_slice_index, max_class_slice, max_voxel_count

def rescale(img):
    tensor_norm = (img - img.min()) / (img.max() - img.min())
    tensor_scaled = tensor_norm * 255
    tensor_final = tensor_scaled.round().astype(np.uint8)
    return tensor_final

# image: feature_prediction, masks: gt_seg_map, seg: seg_prediction
def _add_image_wandb(images, masks, atts, seg=None, mode='train'):
        ## masks of shape (H,W) and value [0,1,2,3,...]
        # masks : segmentation to overaly on the image
        # if dist.get_rank() == 0:
        results = []
        ## Iterate over all images
        for idx in range(images.shape[0]):
            if masks is not None:
                mask_img = wandb.Image(
                    images[idx], masks={
                        # "predictions": {"mask_data": att},
                        "ground_truth": { "mask_data": masks[idx]}
                    },
                    caption="ground-truth"
                )
                results.append(mask_img)

            att  = wandb.Image(rescale(atts[idx]), caption="grounding")
            results.append(att)
            
            if seg is not None:
                seg_img = wandb.Image(
                    images[idx], masks={
                        # "predictions": {"mask_data": att},
                        "ground_truth": { "mask_data": seg[idx]}
                    },
                    caption="segmentation map"
                )
                results.append(seg_img)
            
            
        wandb.log({f"{mode}/grounding": results})


# def select():
#     bs = 12
#     if i % 10000 == 0:
#         mode='train'
#         select_idx = (labels != 0).squeeze()
#         images_ = images[select_idx].cpu()[:bs].permute(0, 2, 3, 1)
#         masks_ = masks[select_idx].cpu()[:bs].squeeze()
#         atts_ = atts[select_idx].detach().cpu()[:bs]
#         seg_ = segs[select_idx].detach().cpu()[:bs].squeeze(1)
#         if recon is not None:
#             recon_ = recon.detach().cpu()[:bs]
#         else:
#             recon_ = None
#         if not config.get('no_wandb', True):
#             images_, masks_, atts_, seg_ = images_.numpy(), masks_.numpy(), atts_.numpy(), seg_.numpy()
#             if recon_ is not None:
#                 recon_ = recon_.numpy()
#             metric_logger._add_image_wandb(images_, masks_, atts_, seg=seg_, recon_pixels=recon_, mode=mode)



def pca_compress(data): # NOTE use for features
    B, H, W, C = data.shape
    # Reshape the tensor to apply PCA: (B*H*W, 256)
    data_flat = data.view(-1, C).numpy()

    # Perform PCA, reducing 256 dimensions to 3
    pca = PCA(n_components=3)
    principalComponents = pca.fit_transform(data_flat)

    # Apply MinMax scaling
    mm = MinMaxScaler()
    mm_data = mm.fit_transform(principalComponents)

    # Reshape back to original spatial dimensions (B, H, W, 3)
    data = torch.tensor(mm_data.reshape(B, H, W, 3)).permute(0, 3, 1, 2)*255
    return data

# if __name__ == "__main__":
#     fea_map = fea_map.view(B, h, w, -1)
#     rgb_fea = pca_compress(fea_map.cpu())

#     ts.save(rgb_fea, f'rgb_fea.png')