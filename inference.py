import os
import sys
sys.path.append(os.path.join(".."))

import torch
import numpy as np
import tifffile
import glob
from torchvision import transforms
import matplotlib.pyplot as plt

from model import *
from prior_model import *
from transforms import *
from utils import *
from dataset import *


def load(checkpoints_dir, model, epoch, optimizer=[]):

    model_dict = torch.load('%s/model_epoch%04d.pth' % (checkpoints_dir, epoch))

    print('Loaded %dth network' % epoch)

    model.load_state_dict(model_dict['model'])
    optimizer.load_state_dict(model_dict['optimizer'])

    return model, optimizer, epoch

def main():

    #********************************************************#

    # project_dir = os.path.join('Z:\\', 'members', 'Rauscher', 'projects', 'OCM_denoising-n2n_training')
    project_dir = os.path.join('C:\\', 'Users', 'rausc', 'Documents', 'EMBL', 'projects', 'adaptive_FastDVDNet')
    data_dir = os.path.join('C:\\', 'Users', 'rausc', 'Documents', 'EMBL', 'data', 'only_three_dataset', 'good_sample-unidentified')
    name = 'only_two_dataset-test_1'
    inference_name = 'inference_140-good_sample-unidentified'
    adaptive_load_epoch = 140
    prior_load_epoch = 120


    #********************************************************#

    results_dir = os.path.join(project_dir, name, 'adaptive', 'results')
    adaptive_checkpoints_dir = os.path.join(project_dir, name, 'adaptive', 'checkpoints')
    prior_checkpoints_dir = os.path.join(project_dir, name, 'prior', 'checkpoints')

    # Make a folder to store the inference
    inference_folder = os.path.join(results_dir, inference_name)
    os.makedirs(inference_folder, exist_ok=True)



    # check if we have a gpu
    device = get_device()

    mean, std = load_normalization_params(adaptive_checkpoints_dir)
    
    inf_transform = transforms.Compose([
        Normalize(mean, std),
        CropToMultipleOf16(),
        ToTensor(),
    ])

    inv_inf_transform = transforms.Compose([
        BackTo01Range(),
        ToNumpy()
    ])

    inf_dataset = DatasetLoadAll(
        data_dir,
        transform=inf_transform
    )

    batch_size = 8
    print("Dataset size:", len(inf_dataset))
    inf_loader = torch.utils.data.DataLoader(
        inf_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )


    adaptive_model = AdaptiveFastDVDnet()
    adaptive_optimizer = torch.optim.Adam(adaptive_model.parameters(), lr=1e-3)
    adaptive_model, _, _ = load(adaptive_checkpoints_dir, adaptive_model, adaptive_load_epoch, adaptive_optimizer)

    prior_model = PriorUNet()
    prior_optimizer = torch.optim.Adam(prior_model.parameters(), lr=1e-3)
    prior_model, _, _ = load(prior_checkpoints_dir, prior_model, prior_load_epoch, prior_optimizer)

    adapt_style = StyleAdaptation()


    print("starting inference")

    with torch.no_grad():

        adaptive_model.eval()
        prior_model.eval()

        # Initialize list to store numpy arrays for output images
        output_images = []

        for batch, data in enumerate(inf_loader):

            input_stack = data[0].to(device)

            prior = prior_model(input_stack)

            adapted_prior = adapt_style(prior)

            output_img = adaptive_model(input_stack, adapted_prior)

            output_img_np = inv_inf_transform(output_img)

            for img in output_img_np:
                output_images.append(img)

            print(f'BATCH {batch+1}/{len(inf_loader)}')

    # Clip output images to the 0-1 range
    output_images_clipped = [np.clip(img, 0, 1) for img in output_images]

    # Stack and save output images
    output_stack = np.stack(output_images_clipped, axis=0).squeeze(-1)  # Remove channel dimension if single channel
    filename = f'output_stack-{name}-{inference_name}.TIFF'
    tifffile.imwrite(os.path.join(inference_folder, filename), output_stack)

    print("Output TIFF stack created successfully.")


if __name__ == '__main__':
    main()


