import time
import os

import numpy as np
import cv2
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T

from typing import List, Dict, Tuple, Union, Final

# module-level variables and constants
OPEN_CV_BLACK: Final[Tuple[int, int, int]] = (0, 0, 0)
OPEN_CV_WHITE: Final[Tuple[int, int, int]] = (255, 255, 255)
OPEN_CV_RED: Final[Tuple[int, int, int]] = (0, 0, 255)
OPEN_CV_BLUE: Final[Tuple[int, int, int]] = (255, 0, 0)
OPEN_CV_GREEN: Final[Tuple[int, int, int]] = (0, 195, 0)
OPEN_CV_BRIGHT_GREEN: Final[Tuple[int, int, int]] = (0, 255, 0)
OPEN_CV_MAGENTA: Final[Tuple[int, int, int]] = (255, 0, 255)
OPEN_CV_YELLOW: Final[Tuple[int, int, int]] = (0, 255, 255)
OPEN_CV_CYAN: Final[Tuple[int, int, int]] = (255, 255, 0)
OPEN_CV_ORANGE: Final[Tuple[int, int, int]] = (0, 165, 255)
OPEN_CV_GRAY: Final[Tuple[int, int, int]] = (169, 169, 169)
GREEN_COLOR_MULTIPLIER: Final[Tuple[float, float, float]] = (0.6, 1.0, 0.6)

# all colors in a specific order for debug image coloring
COLORS: Final[List[Tuple[int, int, int]]] = [
    OPEN_CV_BLACK,
    OPEN_CV_GREEN,
    OPEN_CV_BLUE,
    OPEN_CV_RED,
    OPEN_CV_BRIGHT_GREEN,
    OPEN_CV_MAGENTA,
    OPEN_CV_CYAN,
    OPEN_CV_YELLOW,
    OPEN_CV_WHITE,]

def to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """
    A function to convert a torch input to numpy array.
    Args:
        tensor (torch tensor).
    Returns:
        Converted to numpy array.
    """
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


def show_sample(idx: int, dataset):
    
    img_t, mask_t = dataset[idx]
   
    
    image: np.ndarray = to_numpy(img_t.permute(1, 2, 0).squeeze())
    mask: np.ndarray = to_numpy(mask_t).astype(np.uint8)
    
    # scale back and add the mean, scale to 0-255
    image = ((image * dataset.std + dataset.mean) * 255).astype(np.uint8)
    
    class_ids: List[int] = np.unique(mask)[1:]
    if len(image.shape) < 3:
        image = np.repeat(np.expand_dims(image, axis=2), 3, axis=2)
    
    mask = np.repeat(np.expand_dims(mask, axis=2), 3, axis=2)
    
    for class_id in class_ids:
        class_pixels: Tuple[np.ndarray, np.ndarray] = np.where(mask==class_id)
        image[class_pixels] = 0.6 * image[class_pixels] + 0.4 / max(class_ids) * (mask * COLORS[class_id])[class_pixels]
    
    return image.astype(np.uint8)
    
def pixel_accuracy(output: torch.Tensor, mask: torch.Tensor) -> float:
    """
    Pixel-wise accuracy
    
    Args:
        output: BATCH_SIZE * NUM_CLASSES * H * W tensor of class predictions (logits) for each pixel
        mask: BATCH_SIZE * H * W tensor of semantic mask
    Returns:
        Accuracy
    """
    with torch.no_grad():
        predicted_masks: torch.Tensor = torch.argmax(torch.nn.functional.softmax(output, dim=1), dim=1)
    
    return pixel_accuracy_masks(predicted_masks, mask)

def mIoU(output: torch.Tensor, 
         mask: torch.Tensor, 
         num_classes: int,
         smooth: float = 1e-10, 
         ignore_index_zero: bool = False
        ) -> float:
    """
    Mean IoU (among different classes)
    
    Args:
        output: BATCH_SIZE * NUM_CLASSES * H * W tensor of class logits for each pixel and class (including bg)
        mask: BATCH_SIZE * H * W tensor of semantic mask
        num_classes: Number of class IDs (including index 0 for bg)
        smooth: A small float to avoid divide by zero
        ignore_index_zero: A flag to exclude class ID 0 (bg)
    Returns:
        Average IoU over all classes
    """

    with torch.no_grad():
        predicted_masks: torch.tensor = torch.argmax(torch.nn.functional.softmax(output, dim=1), dim=1)
        
    return mIoU_masks(predicted_masks, mask, num_classes, smooth, ignore_index_zero)
        
def pixel_accuracy_masks(predicted_masks: torch.Tensor, gt_masks: torch.Tensor) -> float:
    """
    Pixel-wise accuracy given ground-truth and predicted masks for a batch
    
    Args:
        predicted_masks: BATCH_SIZE * H * W tensor of class predictions for each pixel
        gt_masks: BATCH_SIZE * H * W tensor of ground-truth semantic masks
    Returns:
        Accuracy over the batch
    """
    with torch.no_grad():
        correct: torch.Tensor = torch.eq(predicted_masks, gt_masks).int()
        accuracy = float(correct.sum()) / float(correct.numel())
    return accuracy

def mIoU_masks(predicted_masks: torch.Tensor, 
               gt_masks: torch.Tensor, 
               num_classes: int,
               smooth: float = 1e-10, 
               ignore_index_zero: bool = False
               ) -> float:
    """
    Mean IoU (among different classes) given ground-truth and predicted masks for a batch
    
    Args:
        predicted_masks: BATCH_SIZE * H * W tensor of class predictions for each pixel
        gt_masks: BATCH_SIZE * H * W tensor of ground-truth semantic masks
        num_classes: Number of class IDs (including index 0 for bg)
        smooth: A small float to avoid divide by zero
        ignore_index_zero: A flag to exclude class ID 0 (bg)
    Returns:
        Average IoU over all classes
    """

    with torch.no_grad():
        
        start_idx: int = 0
        if ignore_index_zero:
            start_idx: int = 1
        
        predicted_masks = predicted_masks.contiguous().view(-1)
        gt_masks = gt_masks.contiguous().view(-1)

        iou_per_class = []
        for class_id in range(start_idx, num_classes): #loop per pixel class
            true_class: torch.Tensor = predicted_masks == class_id
            true_label: torch.Tensor = gt_masks == class_id

            if true_label.long().sum().item() == 0:
                # the class does not exist in this mask
                iou_per_class.append(np.nan)
            else:
                intersect = torch.logical_and(true_class, true_label).sum().float().item()
                union = torch.logical_or(true_class, true_label).sum().float().item()

                iou = (intersect + smooth) / (union + smooth)
                iou_per_class.append(iou)
        
        if np.isnan(iou_per_class).all():
            # in case ignore_index_zero is set to True and all the images only include background (index 0) pixels, 
            # iou_per_class will be all np.nan for non-background class IDs
            # return np.nan to ignore this batch
            return np.nan
            
        return np.nanmean(iou_per_class)
        
class SemanticMaskDataset(Dataset):
    
    def __init__(self, 
                 images_path: str, 
                 masks_path: str,
                 mean: np.array, 
                 std: np.array,
                 model_input_size: Tuple[int, int],
                 transform = None,
                ):

        self.images_path: str = images_path
        self.masks_path: str = masks_path
    
        self.img_names: List[str] = os.listdir(images_path)
        self.img_names: List[str] = sorted([f for f in self.img_names if f.strip().split('.')[-1] == 'jpg'])
        
        self.mask_names: List[str] = os.listdir(masks_path)
        self.mask_names: List[str] = sorted([f for f in self.mask_names if f.strip().split('.')[-1] == 'png'])

        img_file_names: List[str] = [".".join(f.strip().split('.')[:-1]) for f in self.img_names]
        mask_file_names: List[str] = [".".join(f.strip().split('.')[:-1]) for f in self.mask_names]

        self.imgs_paths = [os.path.join(images_path, f) for f in self.img_names]
        self.masks_paths = [os.path.join(masks_path, f) for f in self.mask_names]

        self.transform = transform
        self.mean: np.array = mean
        self.std: np.array = std
        self.model_input_size: Tuple[int, int] = model_input_size
        
        if (len(self.mask_names) != len(self.img_names)):
            print("[ERROR] The mask filenames and the image filenames are not consistent! Dataset will not be instantiated correctly!")
            return 

        if any([img_file_names[i] != mask_file_names[i] for i in range(min(len(img_file_names), len(mask_file_names)))]):
            print("[ERROR] The mask filenames and the image filenames are not consistent! Dataset will not be instantiated correctly!")
         
        
    
    def __len__(self) -> int:
        return len(self.imgs_paths)
    
    def __getitem__(self, idx: int) -> (torch.Tensor, torch.Tensor):
        
        image: np.ndarray = cv2.imread(self.imgs_paths[idx], cv2.IMREAD_UNCHANGED)
        if len(image.shape) < 3:
            # the model expects a 3 channel image, 
            image = np.repeat(np.expand_dims(image, axis=2), 3, axis=2)
        
        # semantic mask
        # we are using np.uint8, hence only 255 segments
        mask_name: str = ".".join(self.img_names[idx].strip().split('.')[:-1]) + '.png'
        semantic_mask: np.ndarray = cv2.imread(os.path.join(self.masks_path, mask_name), cv2.IMREAD_UNCHANGED)

        # make sure the training image and mask are of the correct size
        image = cv2.resize(image, dsize=self.model_input_size, interpolation=cv2.INTER_CUBIC)
        semantic_mask = cv2.resize(semantic_mask, dsize=self.model_input_size, interpolation=cv2.INTER_NEAREST)
        
        if self.transform is not None:
            augmented = self.transform(image=image, mask=semantic_mask)
            image = augmented['image']
            semantic_mask = augmented['mask']

        # convert the image (numpy array) to a torch Tensor and normalize it
        convert_normalize_t = T.Compose([T.ToTensor(), T.Normalize(self.mean, self.std)])
        image_tensor: torch.Tensor = convert_normalize_t(image)
        mask_tensor: torch.Tensor = torch.from_numpy(semantic_mask).long()
        
        return image_tensor, mask_tensor
        
        
def train(model, 
          num_classes,
          train_loader, 
          test_loader, 
          ignore_index_zero, # set to true if the model should not be trained on the background pixels 
          optimizer,         # (e.g., the background is not annotated properly)
          lr_scheduler,
          num_epochs,
          device):
    
    torch.cuda.empty_cache()
    
    # losses, accuracies and mean IoUs over the training epochs
    train_losses: List[float] = []
    test_losses: List[float] = []
    train_ious: List[float] = []
    train_accs: List[float] = []
    test_ious: List[float] = [] 
    test_accs: List[float] = []
    
    # learning rates used for each step (not each epoch as we may use OneCyle scheduling)
    lrs: List[float] = []
    min_loss: float = np.inf

   
    if ignore_index_zero:
        criterion = torch.nn.CrossEntropyLoss(ignore_index=0)
    else:
        criterion = torch.nn.CrossEntropyLoss()
    
    model = model.to(device)
    
    start_time = time.time()
    
    for epoch in range(num_epochs):
        
        since = time.time()
        
        running_loss: float = 0
        iou_score: float = 0
        accuracy: float = 0

        num_valid_train_batches: int = 0
        
        # training loop
        model.train()
        for i, data in enumerate(tqdm(train_loader)):
            # training phase
            images_batch, masks_batch = data    
            images_batch = images_batch.to(device) 
            masks_batch = masks_batch.to(device)

            if ignore_index_zero:
                unique_mask_values_tensor = masks_batch.unique()
                if len(unique_mask_values_tensor) == 1 and unique_mask_values_tensor[0].item() == 0:
                    # the cross entropy loss ignores index 0 and if the mask does not have any non-zero (non-background) pixels, 
                    # the loss will be nan, so we skip this batch
                    # the training dataset is prepared to only contain images with some non-bg pixels, but this step is included 
                    # for sanity
                    # no need to reset the gradients (yet) as we have not taken any forward step
                    continue
                
            
            # forward
            output = model(images_batch)['out']
            loss = criterion(output, masks_batch)
            
            # evaluate metrics
            # mean IoU can only be np.nan if the mask only include background (index 0) pixles and the mIoU function is called
            # to ignore this index (default) 
            # this should never happen, 
            iou_score += mIoU(output, masks_batch, num_classes=num_classes, ignore_index_zero=ignore_index_zero)
            accuracy += pixel_accuracy(output, masks_batch)
            num_valid_train_batches += 1
                
            # backward
            loss.backward()
            optimizer.step() # update weight          
            optimizer.zero_grad() # reset gradient
            
            # update the learning rate only after one batch in case of One-Cycle LR scheduler
            lrs.append(lr_scheduler.get_last_lr()[0])
            if isinstance(lr_scheduler, torch.optim.lr_scheduler.OneCycleLR):
                lr_scheduler.step() 

            # similarly, loss will never be nan
            running_loss += loss.item()
        
        # update the learning rate after one full epoch if LR step scheduler is used
        if isinstance(lr_scheduler, torch.optim.lr_scheduler.StepLR):
            lr_scheduler.step() 
        
        # calculatio mean for all training samples (over number of batches)
        running_loss /= num_valid_train_batches
        iou_score /= num_valid_train_batches
        accuracy /= num_valid_train_batches
        
        # run the validation after each training epoch
        test_iou_score, test_accuracy, test_running_loss = evaluate(
            model, 
            num_classes, 
            test_loader, 
            device, 
            return_loss=True, 
            ignore_index_zero=ignore_index_zero
        )
                       
        # save the results
        train_losses.append(running_loss)
        train_ious.append(iou_score)
        train_accs.append(accuracy)
        
        test_losses.append(test_running_loss)
        test_ious.append(test_iou_score)
        test_accs.append(test_accuracy)
        
        print('saving the model ...')
        torch.save(model.state_dict(), os.path.join(MODEL_PATH, 'checkpoint_' + str(epoch) +'.pt'))
                    
        
        print("Epoch:{}/{} ... \n".format(epoch + 1, num_epochs),
              "Train Loss: {:.3f} \n".format(running_loss),
              "Test Loss: {:.3f} \n".format(test_running_loss),
              "Train mean IoU: {:.3f} \n".format(iou_score),
              "Test mean IoU: {:.3f} \n".format(test_iou_score),
              "Train Accuracy: {:.3f} \n".format(accuracy),
              "Test Accuracy: {:.3f} \n".format(test_accuracy),
              "Time: {:.2f} m".format((time.time() - since) / 60))
        
    history = {'train_loss' : train_losses, 'test_loss': test_losses,
               'train_mean_iou' :train_ious, 'test_mean_iou': test_ious,
               'train_acc': train_accs, 'val_acc': test_accs,
               'lrs': lrs}
    print('Total time: {:.2f} m' .format((time.time()- start_time) / 60))
    return history
    
    
def evaluate(model, 
             num_classes, 
             data_loader, 
             device, 
             return_loss=False, 
             ignore_index_zero=True):
    
    # run the validation
    model.eval()
    model.to(device)
    
    iou_score: float = 0
    accuracy: float = 0
    loss: float = 0
    num_valid_batches: int = 0

    if return_loss:
        if ignore_index_zero:
            criterion = torch.nn.CrossEntropyLoss(ignore_index=0)
        else:
            criterion = torch.nn.CrossEntropyLoss()
    
    # validation loop
    with torch.no_grad():
        for i, data in enumerate(tqdm(data_loader)):  
            # predictions
            images_batch, masks_batch = data    
            images_batch = images_batch.to(device) 
            masks_batch = masks_batch.to(device)

            if ignore_index_zero:
                unique_mask_values_tensor = masks_batch.unique()
                if len(unique_mask_values_tensor) == 1 and unique_mask_values_tensor[0].item() == 0:
                    continue
                
            output = model(images_batch)['out']
                
            # evaluation metrics
            iou_score += mIoU(output, masks_batch, num_classes=num_classes, ignore_index_zero=ignore_index_zero)
            accuracy += pixel_accuracy(output, masks_batch)
                
            # loss 
            if return_loss:
                running_loss += criterion(output, masks_batch).item()                                 
            
            # increament the number of valid batches
            num_valid_batches += 1
    
    iou_score /= num_valid_batches
    accuracy /= num_valid_batches
    if return_loss:
        loss /= num_valid_batches
        return iou_score, accuracy, loss
        
    return iou_score, accuracy
    
