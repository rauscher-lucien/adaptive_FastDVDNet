import os
import numpy as np
import torch        
import tifffile
import matplotlib.pyplot as plt

from utils import *
        


class DatasetLoadAll(torch.utils.data.Dataset):
    def __init__(self, root_folder_path, transform=None):
        """
        Initializes the dataset with the path to a folder and its subfolders containing TIFF stacks, 
        and an optional transform to be applied to each input-target pair.

        Parameters:
        - root_folder_path: Path to the root folder containing TIFF stack files.
        - transform: Optional transform to be applied to each input-target pair.
        """
        self.root_folder_path = root_folder_path
        self.transform = transform
        self.preloaded_data = {}  # To store preloaded data
        self.pairs, self.cumulative_slices = self.preload_and_process_stacks()

    def preload_and_process_stacks(self):
        pairs = []
        cumulative_slices = [0]
        for subdir, _, files in os.walk(self.root_folder_path):
            sorted_files = sorted([f for f in files if f.lower().endswith(('.tif', '.tiff'))])
            for filename in sorted_files:
                full_path = os.path.join(subdir, filename)
                stack = tifffile.imread(full_path)
                self.preloaded_data[full_path] = stack  # Preload data here
                pairs.extend(self.process_stack(full_path, stack))
                cumulative_slices.append(len(pairs))
        return pairs, cumulative_slices

    def process_stack(self, filepath, stack):
        pairs = []
        if stack.shape[0] >= 5:  # Ensure there are enough slices
            for i in range(stack.shape[0] - 4):  # Iterate to form groups of 5 adjacent slices
                input_slices_indices = [i, i+1, i+3, i+4]
                target_slice_index = i+2
                pairs.append((filepath, input_slices_indices, target_slice_index))
        return pairs

    def __len__(self):
        return self.cumulative_slices[-1]

    def __getitem__(self, index):
        # Find which stack the index falls into
        stack_index = next(i for i, total in enumerate(self.cumulative_slices) if total > index) - 1
        # Calculate the local index within that stack's pairs (adjust for the offset)
        if stack_index == 0:
            local_index = index
        else:
            local_index = index - self.cumulative_slices[stack_index]

        file_path, input_slice_indices, target_slice_index = self.pairs[local_index]

        # Access preloaded data instead of reading from file
        volume = self.preloaded_data[file_path]
        input_slices = np.stack([volume[i] for i in input_slice_indices], axis=-1)
        target_slice = volume[target_slice_index][..., np.newaxis]

        if self.transform:
            input_slices, target_slice = self.transform((input_slices, target_slice))

        return input_slices, target_slice
    


class DatasetLoadSingular(torch.utils.data.Dataset):
    def __init__(self, root_folder_path, transform=None):
        """
        Initializes the dataset with paths to files instead of loading all data into memory.
        Each item is loaded and processed on demand.
        """
        self.root_folder_path = root_folder_path
        self.transform = transform
        self.pairs = self.list_all_pairs()

    def list_all_pairs(self):
        pairs = []
        for subdir, _, files in os.walk(self.root_folder_path):
            sorted_files = sorted([f for f in files if f.lower().endswith(('.tif', '.tiff'))])
            for filename in sorted_files:
                full_path = os.path.join(subdir, filename)
                stack_info = self.get_stack_info(full_path)
                pairs.extend(stack_info)
        return pairs

    def get_stack_info(self, filepath):
        """
        Instead of loading the stack, just record the necessary information to load it later.
        """
        with tifffile.TiffFile(filepath) as tif:
            num_frames = len(tif.pages)
            pairs = []
            if num_frames >= 5:
                for i in range(num_frames - 4):
                    input_slice_indices = [i, i+1, i+3, i+4]
                    target_slice_index = i+2
                    pairs.append((filepath, input_slice_indices, target_slice_index))
        return pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        file_path, input_slice_indices, target_slice_index = self.pairs[index]
        stack = tifffile.imread(file_path)

        input_slices = np.stack([stack[i] for i in input_slice_indices], axis=0)[..., np.newaxis]
        target_slice = stack[target_slice_index][..., np.newaxis]

        if self.transform:
            input_slices, target_slice = self.transform((input_slices, target_slice))

        return input_slices, target_slice
    

class PriorDataset(torch.utils.data.Dataset):
    def __init__(self, root_folder_path, transform=None):
        self.root_folder_path = root_folder_path
        self.transform = transform
        self.preloaded_data = {}  # To store preloaded data
        self.pairs, self.cumulative_slices = self.preload_and_make_pairs(root_folder_path)

    def preload_and_make_pairs(self, root_folder_path):
        pairs = []
        cumulative_slices = [0]
        for subdir, _, files in os.walk(root_folder_path):
            sorted_files = sorted([f for f in files if f.lower().endswith(('.tiff', '.tif'))])
            for f in sorted_files:
                full_path = os.path.join(subdir, f)
                volume = tifffile.imread(full_path)
                self.preloaded_data[full_path] = volume  # Preload data here
                num_slices = volume.shape[0]
                if num_slices >= 5:  # Ensure at least 5 slices for forming pairs
                    for i in range(num_slices - 4):
                        input_slices_indices = [i, i+1, i+3, i+4]
                        target_slice_index = i + 2
                        pairs.append((full_path, input_slices_indices, target_slice_index))
                        cumulative_slices.append(cumulative_slices[-1] + 1)
        return pairs, cumulative_slices

    def __len__(self):
        return self.cumulative_slices[-1]

    def __getitem__(self, index):
        pair_index = next(i for i, total in enumerate(self.cumulative_slices) if total > index) - 1
        file_path, input_slice_indices, target_slice_index = self.pairs[pair_index]
        
        # Access preloaded data instead of reading from file
        volume = self.preloaded_data[file_path]
        input_slices = np.stack([volume[i] for i in input_slice_indices], axis=-1)
        target_slice = volume[target_slice_index][..., np.newaxis]

        if self.transform:
            input_slices, target_slice = self.transform((input_slices, target_slice))

        return input_slices, target_slice


