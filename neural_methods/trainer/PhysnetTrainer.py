"""PhysNet Trainer."""
import os
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from evaluation.metrics import calculate_metrics
from neural_methods.loss.PhysNetNegPearsonLoss import Neg_Pearson
from neural_methods.model.PhysNet import PhysNet_padding_Encoder_Decoder_MAX
from neural_methods.trainer.BaseTrainer import BaseTrainer
from torch.autograd import Variable
from tqdm import tqdm
import wandb
import matplotlib.pyplot as plt

class PhysnetTrainer(BaseTrainer):

    def __init__(self, config, data_loader):
        """Inits parameters from args and the writer for TensorboardX."""
        super().__init__()
        self.device = torch.device(config.DEVICE)
        self.max_epoch_num = config.TRAIN.EPOCHS
        self.model_dir = config.MODEL.MODEL_DIR
        self.model_file_name = config.TRAIN.MODEL_FILE_NAME
        self.batch_size = config.TRAIN.BATCH_SIZE
        self.num_of_gpu = config.NUM_OF_GPU_TRAIN
        self.base_len = self.num_of_gpu
        self.config = config
        self.min_valid_loss = None
        self.best_epoch = 0

        self.model = PhysNet_padding_Encoder_Decoder_MAX(
            frames=config.MODEL.PHYSNET.FRAME_NUM).to(self.device)  # [3, T, 128,128]

        if config.TOOLBOX_MODE == "train_and_test":
            self.num_train_batches = len(data_loader["train"])
            self.loss_model = Neg_Pearson()
            self.optimizer = optim.Adam(
                self.model.parameters(), lr=config.TRAIN.LR)
            # See more details on the OneCycleLR scheduler here: https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.OneCycleLR.html
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer, max_lr=config.TRAIN.LR, epochs=config.TRAIN.EPOCHS, steps_per_epoch=self.num_train_batches)
        elif config.TOOLBOX_MODE == "only_test":
            pass
        else:
            raise ValueError("PhysNet trainer initialized in incorrect toolbox mode!")

    def train(self, data_loader):
        """Training routine for model"""
        if data_loader["train"] is None:
            raise ValueError("No data for train")

        for epoch in range(self.max_epoch_num):
            print('')
            print(f"====Training Epoch: {epoch}====")
            running_loss = 0.0
            train_loss = []
            self.model.train()
            tbar = tqdm(data_loader["train"], ncols=80, mode="ascii")
            for idx, batch in enumerate(tbar):
                tbar.set_description("Train epoch %s" % epoch)
                rPPG, x_visual, x_visual3232, x_visual1616 = self.model(
                    batch[0].to(torch.float32).to(self.device))
                BVP_label = batch[1].to(
                    torch.float32).to(self.device)
                rPPG = (rPPG - torch.mean(rPPG)) / torch.std(rPPG)  # normalize
                BVP_label = (BVP_label - torch.mean(BVP_label)) / \
                            torch.std(BVP_label)  # normalize
                loss = self.loss_model(rPPG, BVP_label)
                loss.backward()
                running_loss += loss.item()
                if idx % 100 == 99:  # print every 100 mini-batches
                    print(
                        f'[{epoch}, {idx + 1:5d}] loss: {running_loss / 100:.3f}')
                    running_loss = 0.0
                train_loss.append(loss.item())
                wandb.log({"loss": loss.item()})

                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                tbar.set_postfix(loss=loss.item())
            self.save_model(epoch)
            if not self.config.TEST.USE_LAST_EPOCH: 
                valid_loss = self.valid(data_loader)
                print('validation loss: ', valid_loss)
                wandb.log({"val loss": valid_loss.item()})

                if self.min_valid_loss is None:
                    self.min_valid_loss = valid_loss
                    self.best_epoch = epoch
                    print("Update best model! Best epoch: {}".format(self.best_epoch))
                elif (valid_loss < self.min_valid_loss):
                    self.min_valid_loss = valid_loss
                    self.best_epoch = epoch
                    print("Update best model! Best epoch: {}".format(self.best_epoch))
        if not self.config.TEST.USE_LAST_EPOCH: 
            print("best trained epoch: {}, min_val_loss: {}".format(
                self.best_epoch, self.min_valid_loss))

    def valid(self, data_loader):
        """ Runs the model on valid sets."""
        if data_loader["valid"] is None:
            raise ValueError("No data for valid")

        print('')
        print(" ====Validing===")
        valid_loss = []
        self.model.eval()
        valid_step = 0
        with torch.no_grad():
            vbar = tqdm(data_loader["valid"], ncols=80)
            for valid_idx, valid_batch in enumerate(vbar):
                vbar.set_description("Validation")
                BVP_label = valid_batch[1].to(
                    torch.float32).to(self.device)
                rPPG, x_visual, x_visual3232, x_visual1616 = self.model(
                    valid_batch[0].to(torch.float32).to(self.device))
                rPPG = (rPPG - torch.mean(rPPG)) / torch.std(rPPG)  # normalize
                BVP_label = (BVP_label - torch.mean(BVP_label)) / \
                            torch.std(BVP_label)  # normalize
                loss_ecg = self.loss_model(rPPG, BVP_label)
                valid_loss.append(loss_ecg.item())
                valid_step += 1
                vbar.set_postfix(loss=loss_ecg.item())
            valid_loss = np.asarray(valid_loss)
        return np.mean(valid_loss)

    def test(self, data_loader):
        """ Runs the model on test sets."""
        if data_loader["test"] is None:
            raise ValueError("No data for test")
        
        print('')
        print("===Testing===")
        predictions = dict()
        labels = dict()

        if self.config.TOOLBOX_MODE == "only_test":
            if not os.path.exists(self.config.INFERENCE.MODEL_PATH):
                raise ValueError("Inference model path error! Please check INFERENCE.MODEL_PATH in your yaml.")
            self.model.load_state_dict(torch.load(self.config.INFERENCE.MODEL_PATH))
            print("Testing uses pretrained model!")
            print(self.config.INFERENCE.MODEL_PATH)
        else:
            if self.config.TEST.USE_LAST_EPOCH:
                last_epoch_model_path = os.path.join(
                self.model_dir, self.model_file_name + '_Epoch' + str(self.max_epoch_num - 1) + '.pth')
                print("Testing uses last epoch as non-pretrained model!")
                print(last_epoch_model_path)
                self.model.load_state_dict(torch.load(last_epoch_model_path))
            else:
                best_model_path = os.path.join(
                    self.model_dir, self.model_file_name + '_Epoch' + str(self.best_epoch) + '.pth')
                print("Testing uses best epoch selected using model selection as non-pretrained model!")
                print(best_model_path)
                self.model.load_state_dict(torch.load(best_model_path))

        self.model = self.model.to(self.config.DEVICE)
        self.model.eval()

        with torch.no_grad():
            for batch_cnt, test_batch in enumerate(data_loader['test']):
                batch_size = test_batch[0].shape[0]
                data, label = test_batch[0].to(
                    self.config.DEVICE), test_batch[1].to(self.config.DEVICE)

                # Generate Feature Maps
                self.generate_visualizations(im_data=data, batch_cnt=batch_cnt)

                pred_ppg_test, _, _, _ = self.model(data)
                for idx in range(batch_size):
                    subj_index = test_batch[2][idx]
                    sort_index = int(test_batch[3][idx])
                    if subj_index not in predictions.keys():
                        predictions[subj_index] = dict()
                        labels[subj_index] = dict()
                    predictions[subj_index][sort_index] = pred_ppg_test[idx]
                    labels[subj_index][sort_index] = label[idx]

        print('')
        calculate_metrics(predictions, labels, self.config)

    def save_model(self, index):
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        model_path = os.path.join(
            self.model_dir, self.model_file_name + '_Epoch' + str(index) + '.pth')

        model_path = os.path.abspath(model_path)
        if model_path.startswith(u"\\\\"):
            model_path = u"\\\\?\\UNC\\" + model_path[2:]
        else:
            model_path = u"\\\\?\\" + model_path

        torch.save(self.model.state_dict(), model_path)
        print('Saved Model Path: ', model_path)

    def generate_visualizations(self, im_data, batch_cnt):
        # ============================= FEATURE VISUALIZATION BLOCK ===============================
        model_weights = list()
        conv_layers = list()
        counter = 0

        # from [N, C, D, H, W] to [C, D, H, W]
        im_data = im_data[0, :]

        model_children = list(self.model.children())
        for i in range(len(model_children)):
            if type(model_children[i]) == nn.Conv3d:
                counter += 1
                model_weights.append(model_children[i].weight)
                conv_layers.append(model_children[i])

            elif type(model_children[i]) == nn.Sequential:
                for child in model_children[i].children():
                    if type(child) == nn.Conv3d:
                        counter += 1
                        model_weights.append(child.weight)
                        conv_layers.append(child)

        #print(f"Total Convolutional Layers: {counter}")
        #print("conv_layers")
        layer_out = list()
        names = list()

        # Get Model Layers
        for layer in conv_layers[0:]:
            im_data = layer(im_data)
            layer_out.append(im_data)
            names.append(str(layer))
            pass

        processed = []
        # layer_out = [x.cpu().numpy() for x in layer_out]
        for feature_map in layer_out:
            # Taking the first frame only
            feature_map = feature_map[:, 0, :, :]
            # Squeeze unnecessary as we do not have a batch size dimension
            # feature_map = feature_map.squeeze(0)
            gray_scale = torch.sum(feature_map, 0)
            gray_scale = gray_scale / feature_map.shape[0]
            processed.append(gray_scale.data.cpu().numpy())

        # for fm in processed:
        #    print(fm.shape)

        fig = plt.figure(figsize=(20, 40))
        for map_idx in range(0, len(processed)):
            a = fig.add_subplot(5, 2, map_idx + 1)
            plt.imshow(processed[map_idx])
            a.axis("off")
            a.set_title(f"{names[map_idx].split('(')[0]}_{map_idx + 1}", fontsize=25)

        plt.savefig(f"C:/NIVS Project/UBFC Experiments/FeatureViz/feature_maps_batch_{batch_cnt}.jpg",
                    bbox_inches='tight')
        plt.close()
