import os
import sys
import argparse
sys.path.append(os.path.join(".."))

import logging

class StreamToLogger(object):
    """
    Fake file-like stream object that redirects writes to a logger instance.
    """
    def __init__(self, logger, log_level=logging.INFO):
        self.logger = logger
        self.log_level = log_level
        self.linebuf = ''

    def write(self, buf):
        for line in buf.rstrip().splitlines():
            self.logger.log(self.log_level, line.rstrip())

    def flush(self):
        pass


logging.basicConfig(filename='logging.log',  # Log filename
                    filemode='a',  # Append mode, so logs are not overwritten
                    format='%(asctime)s - %(levelname)s - %(message)s',  # Include timestamp
                    level=logging.INFO,  # Logging level
                    datefmt='%Y-%m-%d %H:%M:%S')  # Timestamp formatlog_file = open('logfile.log', 'w', buffering=1)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)  # Set logging level for console
logging.getLogger('').addHandler(console_handler)

# Redirect stdout and stderr to logging
sys.stdout = StreamToLogger(logging.getLogger('STDOUT'), logging.INFO)
sys.stderr = StreamToLogger(logging.getLogger('STDERR'), logging.ERROR)





from utils import *
from train import *


def main():

    ## parser

    # Check if the script is running on the server by looking for the environment variable
    if os.getenv('RUNNING_ON_SERVER') == 'true':
        # Set up argument parsing
        parser = argparse.ArgumentParser(description='Process data directory.')
        parser.add_argument('--data_dir', type=str, help='Path to the data directory')

        # Parse arguments
        args = parser.parse_args()

        # Now you can use args.data_dir as the path to your data
        data_dir = args.data_dir
        print(f"Using data directory: {data_dir}")
    else:
        # If not running on the server, perhaps use a default data_dir or handle differently
        data_dir = os.path.join('C:\\', 'Users', 'rausc', 'Documents', 'EMBL', 'data', 'only_two_dataset')
        project_dir = os.path.join('C:\\', 'Users', 'rausc', 'Documents', 'EMBL', 'projects', 'adaptive_FastDVDNet')
        project_name = 'only_two_dataset-test_1'
        train_continue = 'on'
        load_epoch = 140
        prior_load_epoch = 120


    data_dict = {}

    data_dict['data_dir'] = data_dir
    data_dict['project_dir'] = project_dir
    data_dict['project_name'] = project_name

    data_dict['num_epoch'] = 300
    data_dict['batch_size'] = 8
    data_dict['lr'] = 1e-4

    data_dict['num_freq_disp'] = 50
    data_dict['num_freq_save'] = 10

    data_dict['train_continue'] = train_continue
    data_dict['load_epoch'] = load_epoch
    data_dict['prior_load_epoch'] = prior_load_epoch



    TRAINER = Trainer(data_dict)
    TRAINER.train()


if __name__ == '__main__':
    main()


