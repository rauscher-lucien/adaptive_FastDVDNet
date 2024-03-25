import os
import torch
import matplotlib.pyplot as plt
import pickle
import time
import logging

from torchvision import transforms

from utils import *
from transforms import *
from dataset import *
from model import *
from prior_model import *


class Trainer:

    def __init__(self, data_dict):

        self.data_dir = data_dict['data_dir']
        self.project_dir = data_dict['project_dir']
        self.project_name = data_dict['project_name']

        self.num_epoch = data_dict['num_epoch']
        self.batch_size = data_dict['batch_size']
        self.lr = data_dict['lr']

        self.num_freq_disp = data_dict['num_freq_disp']
        self.num_freq_save = data_dict['num_freq_save']

        self.train_continue = data_dict['train_continue']
        self.load_epoch = data_dict['load_epoch']
        self.prior_load_epoch = data_dict['prior_load_epoch']

        # create dirs
        self.results_dir, self.checkpoints_dir = create_adaptive_directories(self.project_dir, self. project_name)

        self.adaptive_train_dir = os.path.join(self.results_dir, 'train')
        os.makedirs(self.adaptive_train_dir, exist_ok=True)

        self.prior_checkpoints_dir = os.path.join(self.project_dir, self.project_name, 'prior', 'checkpoints')

        # check if we have a gpu
        self.device = get_device()


    def save(self, checkpoints_dir, model, optimizer, epoch):
        if not os.path.exists(checkpoints_dir):
            os.makedirs(checkpoints_dir)

        torch.save({'model': model.state_dict(),
                    'optimizer': optimizer.state_dict()},
                   '%s/model_epoch%04d.pth' % (checkpoints_dir, epoch))
        

    def load(self, checkpoints_dir, model, epoch, optimizer=[]):

        model_dict = torch.load('%s/model_epoch%04d.pth' % (checkpoints_dir, epoch))

        print('Loaded %dth network' % epoch)

        model.load_state_dict(model_dict['model'])
        optimizer.load_state_dict(model_dict['optimizer'])

        return model, optimizer, epoch
    

    def train(self):

        ### transforms ###

        mean, std = compute_global_mean_and_std(self.data_dir, self.checkpoints_dir)

        transform_train = transforms.Compose([
            Normalize(mean, std),
            RandomCrop(output_size=(64,64)),
            RandomHorizontalFlip(),
            ToTensor()
        ])

        transform_inv_train = transforms.Compose([
            ToNumpy()
        ])


        ### make dataset and loader ###

        ## prepare dataset
        crop_tiff_depth_to_divisible(self.data_dir, self.batch_size)

        dataset_train = DatasetLoadAll(root_folder_path=self.data_dir,
                                    transform=transform_train)

        loader_train = torch.utils.data.DataLoader(
            dataset_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=2)

        num_train = len(dataset_train)
        num_batch_train = int((num_train / self.batch_size) + ((num_train % self.batch_size) != 0))


        ### initialize network ###

        adaptive_model = AdaptiveFastDVDnet()

        criterion = nn.MSELoss(reduction='sum')

        optimizer = torch.optim.Adam(adaptive_model.parameters(), self.lr)

        st_epoch = 0
        if self.train_continue == 'on':
            print(self.checkpoints_dir)
            adaptive_model, optimizer, st_epoch = self.load(self.checkpoints_dir, adaptive_model, self.load_epoch, optimizer)

        ### initialize prior
            
        prior_model = PriorUNet()

        prior_optimizer = torch.optim.Adam(prior_model.parameters(), self.lr)

        prior_model, _, _ = self.load(self.prior_checkpoints_dir, prior_model, self.prior_load_epoch, prior_optimizer)


        for epoch in range(st_epoch + 1, self.num_epoch + 1):

            for batch, data in enumerate(loader_train, 0):

                def should(freq):
                    return freq > 0 and (batch % freq == 0 or batch == num_batch_train)

                adaptive_model.train()
                prior_model.eval()

                optimizer.zero_grad()

                input_stack, target_img = data

                prior = prior_model(input_stack)

                output_img = adaptive_model(input_stack, prior)

                loss = criterion(output_img, target_img)

                loss.backward()
                optimizer.step()

                logging.info('TRAIN: EPOCH %d: BATCH %04d/%04d: LOSS: %.4f'
                             % (epoch, batch, num_batch_train, loss))
                

                if should(self.num_freq_disp):

                    # Convert tensors to numpy arrays
                    input_stack_np = transform_inv_train(input_stack)
                    target_img_np = transform_inv_train(target_img)
                    output_img_np = transform_inv_train(output_img)

                    num_frames = input_stack_np.shape[-1]

                    for j in range(target_img_np.shape[0]):  # Iterate through each item in the batch
                        # Define a base filename that includes epoch, batch, and sample index
                        base_filename = f"sample{j:03d}"

                        # Save each input frame
                        for frame_idx in range(num_frames):
                            input_frame_filename = os.path.join(self.adaptive_train_dir, f"{base_filename}_input_frame{frame_idx}.png")
                            plt.imsave(input_frame_filename, input_stack_np[j, :, :, frame_idx], cmap='gray')

                        # Save the target and output images
                        target_filename = os.path.join(self.adaptive_train_dir, f"{base_filename}_target.png")
                        output_filename = os.path.join(self.adaptive_train_dir, f"{base_filename}_output.png")

                        plt.imsave(target_filename, target_img_np[j, :, :, 0], cmap='gray')
                        plt.imsave(output_filename, output_img_np[j, :, :, 0], cmap='gray')

                        # Optionally, print or log the file paths to verify
                        # print(f"Saved input frames to: {input_frame_filename}")
                        # print(f"Saved target to: {target_filename}")
                        # print(f"Saved output to: {output_filename}")

            if (epoch % self.num_freq_save) == 0:
                self.save(self.checkpoints_dir, adaptive_model, optimizer, epoch)

               
