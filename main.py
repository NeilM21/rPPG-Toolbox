""" The main function of rPPG deep learning pipeline."""

import argparse
import random
import time
import datetime

import numpy as np
import torch
from config import get_config
from dataset import data_loader
from neural_methods import trainer
from unsupervised_methods.unsupervised_predictor import unsupervised_predict
from torch.utils.data import DataLoader
import wandb
import yaml
import pickle

RANDOM_SEED = 100
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# Create a general generator for use with the validation dataloader,
# the test dataloader, and the unsupervised dataloader
general_generator = torch.Generator()
general_generator.manual_seed(RANDOM_SEED)
# Create a training generator to isolate the train dataloader from
# other dataloaders and better control non-deterministic behavior
train_generator = torch.Generator()
train_generator.manual_seed(RANDOM_SEED)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def add_args(parser):
    """Adds arguments for parser."""

    # NIVS - All Train
    #parser.add_argument('--config_file', required=False,
    #                    default="configs/train_configs/NIVS_ALL_PHYSNET_BASIC.yaml", type=str, help="The name of the model.")

    #parser.add_argument('--config_file', required=False,
    #                    default="configs/train_configs/NIVS_ALL_DEEPPHYS_BASIC.yaml", type=str, help="The name of the model.")

    # Pure - NIVS INF
    #parser.add_argument('--config_file', required=False,
    #                    default="configs/train_configs/PURE_PURE_NIVS_PHYSNET_BASIC.yaml", type=str, help="The name of the model.")

    parser.add_argument('--config_file', required=False,
                        default="configs/infer_configs/NIVS_UNSUPERVISED.yaml", type=str, help="The name of the model.")

    # Pure - UBFC INF
    #parser.add_argument('--config_file', required=False,
    #                  default="configs/train_configs/PURE_PURE_UBFC_PHYSNET_BASIC.yaml", type=str, help="The name of the model.")

    '''Neural Method Sample YAMSL LIST:
      SCAMPS_SCAMPS_UBFC_TSCAN_BASIC.yaml
      SCAMPS_SCAMPS_UBFC_DEEPPHYS_BASIC.yaml
      SCAMPS_SCAMPS_UBFC_PHYSNET_BASIC.yaml
      SCAMPS_SCAMPS_PURE_DEEPPHYS_BASIC.yaml
      SCAMPS_SCAMPS_PURE_TSCAN_BASIC.yaml
      SCAMPS_SCAMPS_PURE_PHYSNET_BASIC.yaml
      PURE_PURE_UBFC_TSCAN_BASIC.yaml
      PURE_PURE_UBFC_DEEPPHYS_BASIC.yaml
      PURE_PURE_UBFC_PHYSNET_BASIC.yaml
      PURE_PURE_MMPD_TSCAN_BASIC.yaml
      UBFC_UBFC_PURE_TSCAN_BASIC.yaml
      UBFC_UBFC_PURE_DEEPPHYS_BASIC.yaml
      UBFC_UBFC_PURE_PHYSNET_BASIC.yaml
      MMPD_MMPD_UBFC_TSCAN_BASIC.yaml
    Unsupervised Method Sample YAMSL LIST:
      PURE_UNSUPERVISED.yaml
      UBFC_UNSUPERVISED.yaml
    '''
    return parser


def train_and_test(config, data_loader_dict):
    """Trains the model."""
    if config.MODEL.NAME == "Physnet":
        model_trainer = trainer.PhysnetTrainer.PhysnetTrainer(config, data_loader_dict)
    elif config.MODEL.NAME == "Tscan":
        model_trainer = trainer.TscanTrainer.TscanTrainer(config, data_loader_dict)
    elif config.MODEL.NAME == "EfficientPhys":
        model_trainer = trainer.EfficientPhysTrainer.EfficientPhysTrainer(config, data_loader_dict)
    elif config.MODEL.NAME == 'DeepPhys':
        model_trainer = trainer.DeepPhysTrainer.DeepPhysTrainer(config, data_loader_dict)
    else:
        raise ValueError('Your Model is Not Supported  Yet!')

    # If a more specific config log is desired (currently tested with PhysNet)
    #run_config = set_wandb_parameters(yaml_config=config)
    #wandb.init(project="rPPGToolbox-NIVSDiagnostic-PhysNet", entity="nivs-uom",
    #           name=f"{config.TRAIN.MODEL_FILE_NAME}", config=config)

    wandb.init(project="rPPGToolbox-NIVSDiagnostic-PhysNet", entity="nivs-uom",
               name=f"{config.TRAIN.MODEL_FILE_NAME}", config=config)
    model_trainer.train(data_loader_dict)
    model_trainer.test(data_loader_dict)
    wandb.finish()


def test(config, data_loader_dict):
    """Tests the model."""
    if config.MODEL.NAME == "Physnet":
        model_trainer = trainer.PhysnetTrainer.PhysnetTrainer(config, data_loader_dict)
    elif config.MODEL.NAME == "Tscan":
        model_trainer = trainer.TscanTrainer.TscanTrainer(config, data_loader_dict)
    elif config.MODEL.NAME == "EfficientPhys":
        model_trainer = trainer.EfficientPhysTrainer.EfficientPhysTrainer(config, data_loader_dict)
    elif config.MODEL.NAME == 'DeepPhys':
        model_trainer = trainer.DeepPhysTrainer.DeepPhysTrainer(config, data_loader_dict)
    else:
        raise ValueError('Your Model is Not Supported  Yet!')
    model_trainer.test(data_loader_dict)


def unsupervised_method_inference(config, data_loader):
    if not config.UNSUPERVISED.METHOD:
        raise ValueError("Please set unsupervised method in yaml!")

    experiment_name = "NIVS_Unsupervised_FalseFS30_72x72"
    wandb.init(project="rPPG-Toolbox-Unsupervised", entity="nivs-uom",
               name=f"{experiment_name}", config=config)

    metrics_wandb_table_dict = dict()
    datetime_str = '{date:%Y-%m-%d__%H-%M-%S}'.format(date=datetime.datetime.now())

    box_plots = None
    for unsupervised_method in config.UNSUPERVISED.METHOD:
        if unsupervised_method == "POS":
            box_plots = unsupervised_predict(config, data_loader, "POS", datetime_str=datetime_str,
                                             box_plot_grp=box_plots, eval_table=metrics_wandb_table_dict)
        elif unsupervised_method == "CHROM":
            box_plots = unsupervised_predict(config, data_loader, "CHROM", datetime_str=datetime_str,
                                             box_plot_grp=box_plots, eval_table=metrics_wandb_table_dict)
        elif unsupervised_method == "ICA":
            box_plots = unsupervised_predict(config, data_loader, "ICA", datetime_str=datetime_str,
                                             box_plot_grp=box_plots, eval_table=metrics_wandb_table_dict)
        elif unsupervised_method == "GREEN":
            box_plots = unsupervised_predict(config, data_loader, "GREEN", datetime_str=datetime_str,
                                             box_plot_grp=box_plots, eval_table=metrics_wandb_table_dict)
        elif unsupervised_method == "LGI":
            box_plots = unsupervised_predict(config, data_loader, "LGI", datetime_str=datetime_str,
                                             box_plot_grp=box_plots, eval_table=metrics_wandb_table_dict)
        elif unsupervised_method == "PBV":
            box_plots = unsupervised_predict(config, data_loader, "PBV", datetime_str=datetime_str,
                                             box_plot_grp=box_plots, eval_table=metrics_wandb_table_dict)
        else:
            raise ValueError("Not supported unsupervised method!")

    # for metric in config.UNSUPERVISED.METRICS:
    #    metrics_wandb_table_dict[metric]  = wandb.Table(columns=["type", "value", "algorithm"])
    #    wandb.log({"multiline": wandb.plot_table(
    #        "wandb/line/v0", metrics_wandb_table_dict[metric], {"x": "algorithm", "y": "value", "groupKeys": "type"},
    #        {"title": f"{metric} test HR Estimate Evaluation"})
    #    })


def set_wandb_parameters(yaml_config):
    wandb_config_dict = {"config": {}, "train": {}, "valid": {},
                         "test": {}}

    # TRAIN DICT
    wandb_config_dict["train"]["batch_size"] = yaml_config.TRAIN.BATCH_SIZE
    wandb_config_dict["train"]["epochs"] = yaml_config.TRAIN.EPOCHS
    wandb_config_dict["train"]["lr"] = yaml_config.TRAIN.LR
    wandb_config_dict["train"]["model_file_name"] = yaml_config.TRAIN.MODEL_FILE_NAME
    wandb_config_dict["train"]["fs"] = yaml_config.TRAIN.DATA.FS
    wandb_config_dict["train"]["dataset_name"] = yaml_config.TRAIN.DATA.DATASET
    wandb_config_dict["train"]["do_preprocess"] = yaml_config.TRAIN.DATA.DO_PREPROCESS
    wandb_config_dict["train"]["data_format"] = yaml_config.TRAIN.DATA.DATA_FORMAT
    wandb_config_dict["train"]["data_range"] = f"{yaml_config.TRAIN.DATA.BEGIN} - {yaml_config.TRAIN.DATA.END}"
    wandb_config_dict["train"]["preprocess_data_type"] = yaml_config.TRAIN.DATA.PREPROCESS.DATA_TYPE
    wandb_config_dict["train"]["preprocess_label_type"] = yaml_config.TRAIN.DATA.PREPROCESS.LABEL_TYPE
    wandb_config_dict["train"]["preprocess_do_chunk"] = yaml_config.TRAIN.DATA.PREPROCESS.DO_CHUNK
    wandb_config_dict["train"]["preprocess_chunk_length"] = yaml_config.TRAIN.DATA.PREPROCESS.CHUNK_LENGTH
    wandb_config_dict["train"]["preprocess_dynamic_detection"] = yaml_config.TRAIN.DATA.PREPROCESS.DYNAMIC_DETECTION
    wandb_config_dict["train"]["preprocess_dynamic_detection_frequency"] = yaml_config.TRAIN.DATA.PREPROCESS.DYNAMIC_DETECTION_FREQUENCY
    wandb_config_dict["train"]["preprocess_crop_face"] = yaml_config.TRAIN.DATA.PREPROCESS.CROP_FACE
    wandb_config_dict["train"]["preprocess_large_face_box"] = yaml_config.TRAIN.DATA.PREPROCESS.LARGE_FACE_BOX
    wandb_config_dict["train"]["preprocess_large_box_coef"] = yaml_config.TRAIN.DATA.PREPROCESS.LARGE_BOX_COEF
    wandb_config_dict["train"]["preprocess_data_h_w"] = (yaml_config.TRAIN.DATA.PREPROCESS.H, yaml_config.TRAIN.DATA.PREPROCESS.W)

    # VALID DICT
    wandb_config_dict["valid"]["fs"] = yaml_config.VALID.DATA.FS
    wandb_config_dict["valid"]["dataset_name"] = yaml_config.VALID.DATA.DATASET
    wandb_config_dict["valid"]["do_preprocess"] = yaml_config.VALID.DATA.DO_PREPROCESS
    wandb_config_dict["valid"]["data_format"] = yaml_config.VALID.DATA.DATA_FORMAT
    wandb_config_dict["valid"]["data_range"] = f"{yaml_config.VALID.DATA.BEGIN} - {yaml_config.VALID.DATA.END}"
    wandb_config_dict["valid"]["preprocess_data_type"] = yaml_config.VALID.DATA.PREPROCESS.DATA_TYPE
    wandb_config_dict["valid"]["preprocess_label_type"] = yaml_config.VALID.DATA.PREPROCESS.LABEL_TYPE
    wandb_config_dict["valid"]["preprocess_do_chunk"] = yaml_config.VALID.DATA.PREPROCESS.DO_CHUNK
    wandb_config_dict["valid"]["preprocess_chunk_length"] = yaml_config.VALID.DATA.PREPROCESS.CHUNK_LENGTH
    wandb_config_dict["valid"]["preprocess_dynamic_detection"] = yaml_config.VALID.DATA.PREPROCESS.DYNAMIC_DETECTION
    wandb_config_dict["valid"]["preprocess_dynamic_detection_frequency"] = yaml_config.VALID.DATA.PREPROCESS.DYNAMIC_DETECTION_FREQUENCY
    wandb_config_dict["valid"]["preprocess_crop_face"] = yaml_config.VALID.DATA.PREPROCESS.CROP_FACE
    wandb_config_dict["valid"]["preprocess_large_face_box"] = yaml_config.VALID.DATA.PREPROCESS.LARGE_FACE_BOX
    wandb_config_dict["valid"]["preprocess_large_box_coef"] = yaml_config.VALID.DATA.PREPROCESS.LARGE_BOX_COEF
    wandb_config_dict["valid"]["preprocess_data_h_w"] = (yaml_config.VALID.DATA.PREPROCESS.H, yaml_config.VALID.DATA.PREPROCESS.W)

    # TEST DICT
    wandb_config_dict["test"]["metrics"] = yaml_config.TEST.METRICS
    wandb_config_dict["test"]["use_last_epoch"] = yaml_config.TEST.USE_LAST_EPOCH
    wandb_config_dict["test"]["fs"] = yaml_config.TEST.DATA.FS
    wandb_config_dict["test"]["dataset_name"] = yaml_config.TEST.DATA.DATASET
    wandb_config_dict["test"]["do_preprocess"] = yaml_config.TEST.DATA.DO_PREPROCESS
    wandb_config_dict["test"]["data_format"] = yaml_config.TEST.DATA.DATA_FORMAT
    wandb_config_dict["test"]["data_range"] = f"{yaml_config.TEST.DATA.BEGIN} - {yaml_config.TEST.DATA.END}"
    wandb_config_dict["test"]["preprocess_data_type"] = yaml_config.TEST.DATA.PREPROCESS.DATA_TYPE
    wandb_config_dict["test"]["preprocess_label_type"] = yaml_config.TEST.DATA.PREPROCESS.LABEL_TYPE
    wandb_config_dict["test"]["preprocess_do_chunk"] = yaml_config.TEST.DATA.PREPROCESS.DO_CHUNK
    wandb_config_dict["test"]["preprocess_chunk_length"] = yaml_config.TEST.DATA.PREPROCESS.CHUNK_LENGTH
    wandb_config_dict["test"]["preprocess_dynamic_detection"] = yaml_config.TEST.DATA.PREPROCESS.DYNAMIC_DETECTION
    wandb_config_dict["test"]["preprocess_dynamic_detection_frequency"] = yaml_config.TEST.DATA.PREPROCESS.DYNAMIC_DETECTION_FREQUENCY
    wandb_config_dict["test"]["preprocess_crop_face"] = yaml_config.TEST.DATA.PREPROCESS.CROP_FACE
    wandb_config_dict["test"]["preprocess_large_face_box"] = yaml_config.TEST.DATA.PREPROCESS.LARGE_FACE_BOX
    wandb_config_dict["test"]["preprocess_large_box_coef"] = yaml_config.TEST.DATA.PREPROCESS.LARGE_BOX_COEF
    wandb_config_dict["test"]["preprocess_data_h_w"] = (yaml_config.TEST.DATA.PREPROCESS.H, yaml_config.TEST.DATA.PREPROCESS.W)

    # CONFIG FICT
    wandb_config_dict["config"]["model_drop_rate"] = yaml_config.MODEL.DROP_RATE
    wandb_config_dict["config"]["model_type"] = yaml_config.MODEL.NAME

    if yaml_config.MODEL.NAME.lower() == "physnet":
        wandb_config_dict["config"]["physnet_frame_num"] = yaml_config.MODEL.PHYSNET.FRAME_NUM

    return wandb_config_dict


if __name__ == "__main__":
    # parse arguments.
    parser = argparse.ArgumentParser()
    parser = add_args(parser)
    parser = trainer.BaseTrainer.BaseTrainer.add_trainer_args(parser)
    parser = data_loader.BaseLoader.BaseLoader.add_data_loader_args(parser)
    args = parser.parse_args()

    # configurations.
    config = get_config(args)
    print('Configuration:')
    print(config, end='\n\n')

    data_loader_dict = dict()
    if config.TOOLBOX_MODE == "train_and_test":
        # neural method dataloader
        # train_loader
        if config.TRAIN.DATA.DATASET == "COHFACE":
            # train_loader = data_loader.COHFACELoader.COHFACELoader
            raise ValueError("Unsupported dataset! Currently supporting UBFC, PURE, MMPD, and SCAMPS.")
        elif config.TRAIN.DATA.DATASET == "UBFC":
            train_loader = data_loader.UBFCLoader.UBFCLoader
        elif config.TRAIN.DATA.DATASET == "PURE":
            train_loader = data_loader.PURELoader.PURELoader
        elif config.TRAIN.DATA.DATASET == "SCAMPS":
            train_loader = data_loader.SCAMPSLoader.SCAMPSLoader
        elif config.TRAIN.DATA.DATASET == "MMPD":
            train_loader = data_loader.MMPDLoader.MMPDLoader
        elif config.TRAIN.DATA.DATASET == "NIVS":
            train_loader = data_loader.NIVSLoader.NIVSLoader
        else:
            raise ValueError("Unsupported dataset! Currently supporting UBFC, PURE, MMPD, and SCAMPS.")

        # valid_loader
        if config.VALID.DATA.DATASET == "COHFACE":
            # valid_loader = data_loader.COHFACELoader.COHFACELoader
            raise ValueError("Unsupported dataset! Currently supporting UBFC, PURE, MMPD, and SCAMPS.")
        elif config.VALID.DATA.DATASET == "UBFC":
            valid_loader = data_loader.UBFCLoader.UBFCLoader
        elif config.VALID.DATA.DATASET == "PURE":
            valid_loader = data_loader.PURELoader.PURELoader
        elif config.VALID.DATA.DATASET == "SCAMPS":
            valid_loader = data_loader.SCAMPSLoader.SCAMPSLoader
        elif config.VALID.DATA.DATASET == "MMPD":
            valid_loader = data_loader.MMPDLoader.MMPDLoader
        elif config.VALID.DATA.DATASET == "NIVS":
            valid_loader = data_loader.NIVSLoader.NIVSLoader
        elif config.VALID.DATA.DATASET is None and not config.TEST.USE_LAST_EPOCH:
                raise ValueError("Validation dataset not specified despite USE_LAST_EPOCH set to False!")
        else:
            raise ValueError("Unsupported dataset! Currently supporting UBFC, PURE, MMPD, and SCAMPS.")      

        # Create and initialize the train dataloader given the correct toolbox mode,
        # a supported dataset name, and a valid dataset path
        if (config.TRAIN.DATA.DATASET and config.TRAIN.DATA.DATA_PATH):

            train_data_loader = train_loader(
                name="train",
                data_path=config.TRAIN.DATA.DATA_PATH,
                config_data=config.TRAIN.DATA)
            data_loader_dict['train'] = DataLoader(
                dataset=train_data_loader,
                num_workers=2,
                batch_size=config.TRAIN.BATCH_SIZE,
                shuffle=False,
                worker_init_fn=seed_worker,
                generator=train_generator
            )
        else:
            data_loader_dict['train'] = None

        """for batch_cnt, sample_batch in enumerate(data_loader_dict['train']):
            if batch_cnt == 0:
                data, label = sample_batch[0].numpy(), sample_batch[1].numpy()
                first_vals = label[0, :]
                second_vals=label[1, :]

            latest_data, latest_label = sample_batch[0].numpy(), sample_batch[1].numpy()
            penult_vals = latest_label[0, :]
            ult_vals = latest_label[1, :]
        """
        print()

        # Create and initialize the valid dataloader given the correct toolbox mode,
        # a supported dataset name, and a valid dataset path
        if (config.VALID.DATA.DATASET and config.VALID.DATA.DATA_PATH and not config.TEST.USE_LAST_EPOCH):
            valid_data = valid_loader(
                name="valid",
                data_path=config.VALID.DATA.DATA_PATH,
                config_data=config.VALID.DATA)
            data_loader_dict["valid"] = DataLoader(
                dataset=valid_data,
                num_workers=1,
                batch_size=config.TRAIN.BATCH_SIZE,  # batch size for val is the same as train
                shuffle=False,
                worker_init_fn=seed_worker,
                generator=general_generator
            )
        else:
            data_loader_dict['valid'] = None

    if config.TOOLBOX_MODE == "train_and_test" or config.TOOLBOX_MODE == "only_test":
        # test_loader
        if config.TEST.DATA.DATASET == "COHFACE":
            # test_loader = data_loader.COHFACELoader.COHFACELoader
            raise ValueError("Unsupported dataset! Currently supporting UBFC, PURE, MMPD, and SCAMPS.")
        elif config.TEST.DATA.DATASET == "UBFC":
            test_loader = data_loader.UBFCLoader.UBFCLoader
        elif config.TEST.DATA.DATASET == "PURE":
            test_loader = data_loader.PURELoader.PURELoader
        elif config.TEST.DATA.DATASET == "SCAMPS":
            test_loader = data_loader.SCAMPSLoader.SCAMPSLoader
        elif config.TEST.DATA.DATASET == "MMPD":
            test_loader = data_loader.MMPDLoader.MMPDLoader
        elif config.TEST.DATA.DATASET == "NIVS":
            test_loader = data_loader.NIVSLoader.NIVSLoader
        else:
            raise ValueError("Unsupported dataset! Currently supporting UBFC, PURE, MMPD, and SCAMPS.")

        if config.TOOLBOX_MODE == "train_and_test" and config.TEST.USE_LAST_EPOCH:
            print("Testing uses last epoch, validation dataset is not required.", end='\n\n')   

        # Create and initialize the test dataloader given the correct toolbox mode,
        # a supported dataset name, and a valid dataset path
        if config.TEST.DATA.DATASET and config.TEST.DATA.DATA_PATH:
            test_data = test_loader(
                name="test",
                data_path=config.TEST.DATA.DATA_PATH,
                config_data=config.TEST.DATA)
            data_loader_dict["test"] = DataLoader(
                dataset=test_data,
                num_workers=1,
                batch_size=config.INFERENCE.BATCH_SIZE,
                shuffle=False,
                worker_init_fn=seed_worker,
                generator=general_generator
            )
        else:
            data_loader_dict['test'] = None

    elif config.TOOLBOX_MODE == "unsupervised_method":
        # unsupervised method dataloader
        if config.UNSUPERVISED.DATA.DATASET == "COHFACE":
            # unsupervised_loader = data_loader.COHFACELoader.COHFACELoader
            raise ValueError("Unsupported dataset! Currently supporting UBFC, PURE, MMPD, and SCAMPS.")
        elif config.UNSUPERVISED.DATA.DATASET == "UBFC":
            unsupervised_loader = data_loader.UBFCLoader.UBFCLoader
        elif config.UNSUPERVISED.DATA.DATASET == "PURE":
            unsupervised_loader = data_loader.PURELoader.PURELoader
        elif config.UNSUPERVISED.DATA.DATASET == "SCAMPS":
            unsupervised_loader = data_loader.SCAMPSLoader.SCAMPSLoader
        elif config.UNSUPERVISED.DATA.DATASET == "MMPD":
            unsupervised_loader = data_loader.MMPDLoader.MMPDLoader
        elif config.UNSUPERVISED.DATA.DATASET == "NIVS":
            unsupervised_loader = data_loader.NIVSLoader.NIVSLoader
        else:
            raise ValueError("Unsupported dataset! Currently supporting UBFC, PURE, MMPD, and SCAMPS.")

        unsupervised_data = unsupervised_loader(
            name="unsupervised",
            data_path=config.UNSUPERVISED.DATA.DATA_PATH,
            config_data=config.UNSUPERVISED.DATA)
        data_loader_dict["unsupervised"] = DataLoader(
            dataset=unsupervised_data,
            num_workers=1,
            batch_size=1,
            shuffle=False,
            worker_init_fn=seed_worker,
            generator=general_generator
        )

    else:
        raise ValueError("Unsupported toolbox_mode! Currently support train_and_test or only_test or unsupervised_method.")

    if config.TOOLBOX_MODE == "train_and_test":
        train_and_test(config, data_loader_dict)
    elif config.TOOLBOX_MODE == "only_test":
        test(config, data_loader_dict)
    elif config.TOOLBOX_MODE == "unsupervised_method":
        unsupervised_method_inference(config, data_loader_dict)
    else:
        print("TOOLBOX_MODE only support train_and_test or only_test !", end='\n\n')
