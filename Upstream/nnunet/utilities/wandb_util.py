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