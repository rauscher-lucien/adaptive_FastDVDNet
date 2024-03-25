import numpy as np
import os
import sys
from skimage import filters, exposure
import matplotlib.pyplot as plt
from PIL import Image
import torch

class Normalize(object):
    """
    Normalize an image using mean and standard deviation.
    
    Args:
        mean (float or tuple): Mean for each channel.
        std (float or tuple): Standard deviation for each channel.
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        """
        Args:
            data (tuple): Containing input and target images to be normalized.
        
        Returns:
            Tuple: Normalized input and target images.
        """
        input_img, target_img = data

        # Normalize input image
        input_normalized = (input_img - self.mean) / self.std

        # Normalize target image
        target_normalized = (target_img - self.mean) / self.std

        return input_normalized, target_normalized


class RandomFlip(object):

    def __call__(self, data):

        input_img, target_img = data

        if np.random.rand() > 0.5:
            input_img = np.fliplr(input_img)
            target_img = np.fliplr(target_img)

        if np.random.rand() > 0.5:
            input_img = np.flipud(input_img)
            target_img = np.flipud(target_img)

        return input_img, target_img
    

class RandomCropVideo(object):
    """
    Randomly crop images from a stack of grayscale slices (input) and a single grayscale slice (target),
    where each slice includes a channel dimension.
    """

    def __init__(self, output_size=(64, 64)):
        """
        Initializes the RandomCrop transformer with the desired output size.

        Parameters:
        - output_size (tuple): The target output size (height, width).
        """
        self.output_size = output_size

    def __call__(self, data):
        """
        Apply random cropping to a stack of input slices and a single target slice.

        Parameters:
        - data (tuple): A tuple containing the input stack and target slice.

        Returns:
        - Tuple: Randomly cropped input stack and target slice.
        """
        input_stack, target_slice = data
        h, w = input_stack.shape[1:3]  # Assuming input_stack shape is (num_slices, H, W, C)
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h + 1)
        left = np.random.randint(0, w - new_w + 1)

        # Crop each slice in the input stack
        cropped_input_stack = input_stack[:, top:top+new_h, left:left+new_w, :]

        # Crop the target slice
        cropped_target_slice = target_slice[top:top+new_h, left:left+new_w, :]

        return cropped_input_stack, cropped_target_slice
    

class RandomCrop:
    def __init__(self, output_size=(64, 64)):
        """
        RandomCrop constructor for cropping both the input stack of slices and the target slice.
        Args:
            output_size (tuple): The desired output size (height, width).
        """
        self.output_size = output_size

    def __call__(self, data):
        """
        Apply the cropping operation.
        Args:
            data (tuple): A tuple containing the input stack and the target slice.
        Returns:
            Tuple: Cropped input stack and target slice.
        """
        input_stack, target_slice = data

        h, w, _ = input_stack.shape
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        input_cropped = input_stack[top:top+new_h, left:left+new_w, :]
        target_cropped = target_slice[top:top+new_h, left:left+new_w, :]

        return input_cropped, target_cropped
    

class RandomHorizontalFlip:
    def __call__(self, data):
        """
        Apply random horizontal flipping to both the input stack of slices and the target slice.
        In 50% of the cases, only horizontal flipping is applied without vertical flipping.
        
        Args:
            data (tuple): A tuple containing the input stack and the target slice.
        
        Returns:
            Tuple: Horizontally flipped input stack and target slice, if applied.
        """
        input_stack, target_slice = data

        # Apply horizontal flipping with a 50% chance
        if np.random.rand() > 0.5:
            # Flip along the width axis (axis 1), keeping the channel dimension (axis 2) intact
            input_stack = np.flip(input_stack, axis=1)
            target_slice = np.flip(target_slice, axis=1)

        # With the modified requirements, we remove the vertical flipping part
        # to ensure that only horizontal flipping is considered.

        return input_stack, target_slice




class CropToMultipleOf16Inference(object):
    """
    Crop an image to ensure its height and width are multiples of 16.
    This is particularly useful for models that require input dimensions to be divisible by certain values.
    """

    def __call__(self, img):
        """
        Args:
            img (numpy.ndarray): Image to be cropped.

        Returns:
            numpy.ndarray: Cropped image.
        """
        h, w = img.shape[:2]  # Assuming img is a numpy array with shape (H, W, C) or (H, W)

        # Compute new dimensions to be multiples of 16
        new_h = h - (h % 16)
        new_w = w - (w % 16)

        # Calculate cropping margins
        top = (h - new_h) // 2
        left = (w - new_w) // 2

        # Generate indices for cropping
        id_y = np.arange(top, top + new_h, 1)[:, np.newaxis].astype(np.int32)
        id_x = np.arange(left, left + new_w, 1).astype(np.int32)

        # Crop the image
        cropped_image = img[id_y, id_x]

        return cropped_image


class CropToMultipleOf16Video(object):
    """
    Crop a stack of images and a single image to ensure their height and width are multiples of 16.
    This is particularly useful for models that require input dimensions to be divisible by certain values.
    """

    def __call__(self, data):
        """
        Args:
            data (tuple): A tuple where the first element is an input stack with shape (4, H, W, 1)
                          and the second element is a target image with shape (H, W, 1).

        Returns:
            tuple: A tuple containing the cropped input stack and target image.
        """
        input_stack, target_img = data

        # Crop the input stack
        cropped_input_stack = [self.crop_image(frame) for frame in input_stack]

        # Crop the target image
        cropped_target_img = self.crop_image(target_img)

        return np.array(cropped_input_stack), cropped_target_img

    def crop_image(self, img):
        """
        Crop a single image to make its dimensions multiples of 16.

        Args:
            img (numpy.ndarray): Single image to be cropped.

        Returns:
            numpy.ndarray: Cropped image.
        """
        h, w = img.shape[:2]  # Assuming img is a numpy array with shape (H, W, C) or (H, W)

        # Compute new dimensions to be multiples of 16
        new_h = h - (h % 16)
        new_w = w - (w % 16)

        # Calculate cropping margins
        top = (h - new_h) // 2
        left = (w - new_w) // 2

        # Crop the image
        cropped_image = img[top:top+new_h, left:left+new_w]

        return cropped_image
    

import torch
import numpy as np

class CropToMultipleOf16(object):
    """
    Crop a tensor representing a stack of images and a single image tensor to ensure their height and width
    are multiples of 16. This is useful for models that require input dimensions to be divisible by certain values.
    """

    def __call__(self, data):
        """
        Args:
            data (tuple): A tuple where the first element is an input stack with shape (H, W, 4)
                          and the second element is a target image with shape (H, W, 1).

        Returns:
            tuple: A tuple containing the cropped input stack and target image tensors.
        """
        input_stack, target_img = data

        # Crop the input stack
        cropped_input_stack = self.crop_image(input_stack)

        # Crop the target image
        cropped_target_img = self.crop_image(target_img)

        return cropped_input_stack, cropped_target_img

    def crop_image(self, img):
        """
        Crop a single tensor to make its dimensions multiples of 16.

        Args:
            img (torch.Tensor): Tensor representing a single image to be cropped.

        Returns:
            torch.Tensor: Cropped image tensor.
        """
        h, w = img.shape[:2]  # Assuming img is a PyTorch tensor with shape (H, W, C)

        # Compute new dimensions to be multiples of 16
        new_h = h - (h % 16)
        new_w = w - (w % 16)

        # Calculate cropping margins
        top = (h - new_h) // 2
        left = (w - new_w) // 2

        # Crop the image tensor
        cropped_image = img[top:top+new_h, left:left+new_w]

        return cropped_image




class SquareCrop(object):
    def __call__(self, data):
        """
        Crop input and target images to form square images.

        Args:
        - data: a tuple containing input and target images as NumPy arrays.

        Returns:
        - A tuple containing cropped input and target images as NumPy arrays.
        """
        input_img, target_img = data

        # Get the minimum side length
        min_side_length = min(input_img.shape[0], input_img.shape[1])

        # Calculate the cropping boundaries
        start_row = (input_img.shape[0] - min_side_length) // 2
        start_col = (input_img.shape[1] - min_side_length) // 2

        # Crop input image
        cropped_input = input_img[start_row:start_row + min_side_length,
                                   start_col:start_col + min_side_length, :]

        # Crop target image
        cropped_target = target_img[start_row:start_row + min_side_length,
                                     start_col:start_col + min_side_length, :]

        return cropped_input, cropped_target




class ToTensorAdaptive(object):
    """
    Convert stacks of images or single images to PyTorch tensors, specifically handling a tuple
    of a stack of input frames and a single target frame for grayscale images.
    The input stack is expected to have the shape (4, H, W, 1) and the target image (H, W, 1).
    It converts them to PyTorch's (B, C, H, W) format for the stack and (C, H, W) for the single image.
    """

    def __call__(self, data):
        """
        Convert a tuple of input stack and target image to PyTorch tensors, adjusting the channel position.
        
        Args:
            data (tuple): A tuple where the first element is an input stack with shape (4, H, W, 1)
                          and the second element is a target image with shape (H, W, 1).
        
        Returns:
            tuple of torch.Tensor: A tuple containing the converted input stack as a PyTorch tensor
                                   with shape (4, 1, H, W) and the target image as a PyTorch tensor
                                   with shape (1, H, W).
        """
        input_stack, target_img = data
        
        # Convert the input stack to tensor and adjust dimensions
        input_stack_tensor = torch.from_numpy(input_stack.transpose(3, 1, 2, 0).astype(np.float32))
        
        # Convert the target image to tensor and adjust dimensions
        target_img_tensor = torch.from_numpy(target_img.transpose(2, 0, 1).astype(np.float32))
        
        return input_stack_tensor, target_img_tensor

    
class ToTensor(object):
    """
    Convert tuples of single images to PyTorch tensors. The input is expected to be a tuple
    of two numpy arrays, each in the format (h, w, c), and it converts them to PyTorch's
    (c, h, w) format.
    """

    def __call__(self, data_tuple):
        """
        Convert a tuple of single images to PyTorch tensors, adjusting the channel position.

        Args:
            data_tuple (tuple of numpy.ndarray): The input should be a tuple of two images,
            each in the format (h, w, c).

        Returns:
            tuple of torch.Tensor: The converted images as PyTorch tensors in the format (c, h, w).
        """
        if not isinstance(data_tuple, tuple) or len(data_tuple) != 2:
            raise TypeError("Input must be a tuple of two numpy.ndarray.")
        
        def convert(image):
            if not isinstance(image, np.ndarray):
                raise TypeError("Each element of the tuple must be a numpy.ndarray.")
            if image.ndim != 3:
                raise ValueError("Unsupported image format: must be (h, w, c).")
            return torch.from_numpy(image.transpose(2, 0, 1).astype(np.float32))
        
        T = tuple(convert(image) for image in data_tuple)

        return T


class ToNumpy(object):
    """
    Convert PyTorch tensors to numpy arrays. This version specifically handles tensors in the format
    (B, C, H, W) and converts them to the format (B, H, W, C), where B is the batch size,
    C is the number of channels, H is the height, and W is the width of the images.
    """
    def __call__(self, tensor):
        """
        Convert a batch of images from a PyTorch tensor in the format (B, C, H, W)
        to a numpy array in the format (B, H, W, C).

        Args:
            tensor (torch.Tensor): A PyTorch tensor in the format (B, C, H, W).

        Returns:
            numpy.ndarray: A numpy array in the format (B, H, W, C).
        """
        if tensor.ndim != 4:  # Ensure tensor is a batch of images
            raise ValueError("Unsupported tensor format: input must be a batch of images (B, C, H, W).")
        
        # Convert tensor to numpy array, adjusting channel position
        return tensor.to('cpu').detach().numpy().transpose(0, 2, 3, 1)





class Denormalize(object):
    """
    Denormalize an image using mean and standard deviation.
    
    Args:
        mean (float or tuple): Mean for each channel used during normalization.
        std (float or tuple): Standard deviation for each channel used during normalization.
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, img):
        """
        Args:
            img: A normalized image (as a numpy array) to be denormalized.
        
        Returns:
            The denormalized image as a numpy array.
        """
        # Denormalize the image
        denormalized_img = (img * self.std) + self.mean
        return denormalized_img



class NormalizeInference(object):
    """
    Normalize an image for inference using mean and standard deviation.

    This is adapted for inference where only one image is processed at a time,
    unlike during training where input-target pairs might be used.

    Args:
        mean (float or tuple): Mean for each channel.
        std (float or tuple): Standard deviation for each channel.
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, img):
        """
        Args:
            img (numpy.ndarray or torch.Tensor): Image to be normalized.

        Returns:
            numpy.ndarray or torch.Tensor: Normalized image.
        """
        # Normalize the image
        normalized_img = (img - self.mean) / self.std
        return normalized_img
    

class MinMaxNormalize(object):
    """
    Normalize images to the 0-1 range using global minimum and maximum values provided at initialization.
    """

    def __init__(self, global_min, global_max):
        """
        Initializes the normalizer with global minimum and maximum values.

        Parameters:
        - global_min (float): The global minimum value used for normalization.
        - global_max (float): The global maximum value used for normalization.
        """
        self.global_min = global_min
        self.global_max = global_max

    def __call__(self, data):
        """
        Normalize input and target images to the 0-1 range using the global min and max.

        Args:
            data (tuple): Containing input and target images to be normalized.

        Returns:
            Tuple: Normalized input and target images.
        """
        input_img, target_img = data

        # Normalize input image
        input_normalized = (input_img - self.global_min) / (self.global_max - self.global_min)
        input_normalized = np.clip(input_normalized, 0, 1)  # Ensure within [0, 1] range

        # Normalize target image
        target_normalized = (target_img - self.global_min) / (self.global_max - self.global_min)
        target_normalized = np.clip(target_normalized, 0, 1)  # Ensure within [0, 1] range

        return input_normalized.astype(np.float32), target_normalized.astype(np.float32)



class MinMaxNormalizeInference(object):
    """
    Normalize an image to the 0-1 range using global minimum and maximum values provided at initialization.
    This is adapted for inference where only one image is processed at a time.
    
    Args:
        global_min (float): The global minimum value used for normalization.
        global_max (float): The global maximum value used for normalization.
    """

    def __init__(self, global_min, global_max):
        self.global_min = global_min
        self.global_max = global_max

    def __call__(self, img):
        """
        Normalize a single image to the 0-1 range using the global min and max.

        Args:
            img (numpy.ndarray): Image to be normalized.

        Returns:
            numpy.ndarray: Normalized image.
        """
        # Normalize the image
        normalized_img = (img - self.global_min) / (self.global_max - self.global_min)
        normalized_img = np.clip(normalized_img, 0, 1)  # Ensure within [0, 1] range
        
        return normalized_img.astype(np.float32)
    

class MinMaxNormalizeVideo(object):
    """
    Normalize images to the 0-1 range using global minimum and maximum values provided at initialization.
    This version is adapted to work with a stack of grayscale slices as input and a single grayscale slice as target,
    where each slice includes a channel dimension.
    """

    def __init__(self, global_min, global_max):
        """
        Initializes the normalizer with global minimum and maximum values.

        Parameters:
        - global_min (float): The global minimum value used for normalization.
        - global_max (float): The global maximum value used for normalization.
        """
        self.global_min = global_min
        self.global_max = global_max

    def __call__(self, data):
        """
        Apply normalization to a stack of input slices and a single target slice.

        Parameters:
        - data (tuple): A tuple containing the input stack and target slice.

        Returns:
        - Tuple: Normalized input stack and target slice.
        """
        input_stack, target_slice = data
        # Normalize each slice in the input stack
        normalized_input_stack = (input_stack - self.global_min) / (self.global_max - self.global_min)
        normalized_input_stack = np.clip(normalized_input_stack, 0, 1)
        
        # Normalize the target slice
        normalized_target_slice = (target_slice - self.global_min) / (self.global_max - self.global_min)
        normalized_target_slice = np.clip(normalized_target_slice, 0, 1)

        return normalized_input_stack, normalized_target_slice


    



class RandomSubsampleBatch(object):
    """
    Subsamples batches of images by applying a 2x2 sliding window across each image in the batch
    and randomly selecting one of the 4 pixels within each window. This process is applied
    independently to batches of input and target images, resulting in subsampled images
    with half the size of the originals.
    """

    def __call__(self, batch_data):
        """
        Subsample batches of input and target images.

        Args:
        - batch_data: a tuple containing two numpy arrays (input_patches, target_patches),
                      each with the shape (batch_size, height, width, channels).

        Returns:
        - A tuple containing subsampled batches of input and target images as NumPy arrays.
        """
        input_patches, target_patches = batch_data

        def subsample_batch(batch):
            subsampled_batch = []
            for img in batch:
                # Determine the size of the subsampled image
                new_height = img.shape[0] // 2
                new_width = img.shape[1] // 2

                # Initialize the subsampled image
                subsampled = np.zeros((new_height, new_width, img.shape[2]), dtype=img.dtype)

                # Apply 2x2 sliding window and randomly select one pixel from each window
                for i in range(new_height):
                    for j in range(new_width):
                        window = img[i*2:(i+1)*2, j*2:(j+1)*2, :]
                        random_row = np.random.randint(0, 2)
                        random_col = np.random.randint(0, 2)
                        subsampled[i, j, :] = window[random_row, random_col, :]
                
                subsampled_batch.append(subsampled)

            return np.array(subsampled_batch)

        # Subsample both input and target batches
        subsampled_input_patches = subsample_batch(input_patches)
        subsampled_target_patches = subsample_batch(target_patches)

        return subsampled_input_patches, subsampled_target_patches


class RandomSubsample(object):
    """
    Subsamples an image by applying a 2x2 sliding window across the image
    and randomly selecting one of the 4 pixels within each window. This process is applied
    independently to both an input and a target image, resulting in subsampled images
    with half the size of the originals.
    """

    def __call__(self, data):
        """
        Subsample an input and a target image.

        Args:
        - data: a tuple containing two numpy arrays (input_image, target_image),
                each with the shape (height, width, channels).

        Returns:
        - A tuple containing subsampled input and target images as NumPy arrays.
        """
        input_image, target_image = data

        def subsample_image(img):
            # Determine the size of the subsampled image
            new_height = img.shape[0] // 2
            new_width = img.shape[1] // 2

            # Initialize the subsampled image
            subsampled = np.zeros((new_height, new_width, img.shape[2]), dtype=img.dtype)

            # Apply 2x2 sliding window and randomly select one pixel from each window
            for i in range(new_height):
                for j in range(new_width):
                    window = img[i*2:(i+1)*2, j*2:(j+1)*2, :]
                    random_row = np.random.randint(0, 2)
                    random_col = np.random.randint(0, 2)
                    subsampled[i, j, :] = window[random_row, random_col, :]
            
            return subsampled

        # Subsample both the input and the target images
        subsampled_input = subsample_image(input_image)
        subsampled_target = subsample_image(target_image)

        return subsampled_input, subsampled_target
    


class BackTo01Range(object):
    """
    Normalize a tensor to the range [0, 1] based on its own min and max values.
    """

    def __call__(self, tensor):
        """
        Args:
            tensor: A tensor with any range of values.
        
        Returns:
            A tensor normalized to the range [0, 1].
        """
        min_val = tensor.min()
        max_val = tensor.max()
        
        # Avoid division by zero in case the tensor is constant
        if (max_val - min_val).item() > 0:
            # Normalize the tensor to [0, 1] based on its dynamic range
            normalized_tensor = (tensor - min_val) / (max_val - min_val)
        else:
            # If the tensor is constant, set it to a default value, e.g., 0, or handle as needed
            normalized_tensor = tensor.clone().fill_(0)  # Here, setting all values to 0

        return normalized_tensor

