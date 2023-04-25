"""
In-house NIVS Experiment Dataset Loader

Project: Non-Contact Imaging for Vital Sign Monitoring (NIVS) Externally funded: MCST-CVP R&I-2018-004-V (2018)

"""
import glob
import os
import re

import cv2
import datetime
from tqdm import tqdm
from multiprocessing import Pool, Process, Value, Array, Manager

import numpy as np
import pandas as pd
from dataset.data_loader.BaseLoader import BaseLoader


class NIVSLoader(BaseLoader):
    def __init__(self, name, data_path, config_data):
        """Initializes a NIVS Dataloader
            Args:
                data_path(str): path of a folder which stores raw video and bvp data.
                e.g. data_path for the raw NIVS data should be "NIVS Data/data" for below dataset structure:
                -----------------
                     data/
                     |   |-- Subject_001/
                     |       |-- S001_R2L_Trimmed.mts
                     |       |-- S001_R2L_groundtruth.csv
                     |   |-- Subject_002/
                     |       |-- S002_R2L_Trimmed.mts
                     |       |-- S002_R2L_groundtruth.csv
                     |...
                     |   |-- Subject_00X/
                     |       |-- S00X_R2L_Trimmed.mts
                     |       |-- S00X_R2L_groundtruth.csv
                -----------------
                name(string): name of the dataloader.
                config_data(CfgNode): data settings(ref:config.py).
        """
        super().__init__(name, data_path, config_data)

    def get_data(self, data_path):
        data_dirs = glob.glob(data_path + os.sep + "Subject*")
        if not data_dirs:
            raise ValueError(self.name + "subject data not found!")
        dirs = [{"index": re.search("Subject (\d+)", data_dir).group(0),
                 "path": data_dir} for data_dir in data_dirs]
        return dirs

    def get_data_subset(self, data_dirs, begin, end):
        """Returns a subset of data dirs, split with begin and end values"""
        if begin == 0 and end == 1: # return the full directory if begin == 0 and end == 1
            return data_dirs

        file_num = len(data_dirs)
        choose_range = range(int(begin * file_num), int(end * file_num))
        data_dirs_new = []

        for i in choose_range:
            data_dirs_new.append(data_dirs[i])

        return data_dirs_new

    def multi_process_manager(self, data_dirs,  config_preprocess, subj_idx=0):
        """ NIVS Dataset Override - In this case the multiprocess manager will be run for each video, and rather
        than processing separate one-minute clips, will store clips of a defined frame cap from a large video"""

        print(f"====================== DATA-LOADING ======================\n")
        saved_filename = data_dirs[subj_idx]['index']
        subject_number = saved_filename.split()[-1]

        subject_video_name = f"S{subject_number}_R2L_Trimmed.mts"
        subject_bvp_name = f"S{subject_number}_R2L_groundtruth.csv"

        subject_video_path = os.path.join(data_dirs[subj_idx]['path'], subject_video_name)
        subject_bvp_path = os.path.join(data_dirs[subj_idx]['path'], subject_bvp_name)

        # Get number of frames since NIVS videos can't be loaded into RAM whole
        print(f"[DATA-LOADING] Processing Video {subject_video_name}...")
        num_frames = self.get_video_metadata(subject_video_path)
        nivs_gt_df = pd.read_csv(subject_bvp_path)

        # Exactly divisible by 128
        frame_load_cap = 1280
        frames_left_over = num_frames % frame_load_cap

        if frames_left_over != 0:
            num_clip_subsets = int(num_frames / frame_load_cap) + 1
        else:
            num_clip_subsets = int(num_frames / frame_load_cap)

        pbar = tqdm(range(num_clip_subsets), ascii=True)

        # shared data resource
        manager = Manager()
        file_list_dict = manager.dict()
        p_list = []
        running_num = 0
        clip_counter = 0

        for i in range(num_clip_subsets):
            process_flag = True
            while process_flag:  # ensure that every i creates a process
                if running_num < 3:  # in case of too many processes
                    clip_start_frame_idx = frame_load_cap * i
                    clip_end_frame_idx = (frame_load_cap * (i + 1)) - 1
                    if i == num_clip_subsets - 1:
                        clip_end_frame_idx = int(clip_start_frame_idx + frames_left_over - 1)
                    p = Process(target=self.preprocess_dataset_subprocess,
                                args=(config_preprocess, i, nivs_gt_df, subject_video_path, clip_start_frame_idx,
                                      clip_end_frame_idx, file_list_dict, saved_filename))
                    p.start()
                    p_list.append(p)
                    clip_counter += 1
                    running_num += 1
                    process_flag = False
                for p_ in p_list:
                    if not p_.is_alive():
                        p_list.remove(p_)
                        p_.join()
                        running_num -= 1
                        pbar.update(1)
        # join all processes
        for p_ in p_list:
            p_.join()
            pbar.update(1)
        pbar.close()

        return file_list_dict

    def preprocess_dataset_subprocess(self, config_preprocess, i, nivs_gt_df, subject_video_path,
                                      clip_start_frame_idx, clip_end_frame_idx, file_list_dict, saved_filename):

        running_clip_count = 0

        frames = self.read_video(video_file=subject_video_path, start_idx=clip_start_frame_idx, end_idx=clip_end_frame_idx)
        bvps = self.read_wave(nivs_gt_df=nivs_gt_df, start_idx=clip_start_frame_idx, end_idx=clip_end_frame_idx)

        frames_clips, bvps_clips = self.preprocess(frames, bvps, config_preprocess, config_preprocess.LARGE_FACE_BOX)
        sub_clip_saved_filename = f"{saved_filename}clip{i}"
        count, input_name_list, label_name_list = self.save_multi_process(frames_clips, bvps_clips,
                                                                          sub_clip_saved_filename)
        running_clip_count += len(bvps_clips)
        file_list_dict[i] = input_name_list
        frames = list()
        bvps = list()
        frames_clips = list()
        bvps_clips = list()



    @staticmethod
    def get_video_metadata(video_file):
        videoCap = cv2.VideoCapture(video_file)
        num_frames = videoCap.get(cv2.CAP_PROP_FRAME_COUNT)
        video_fps = videoCap.get(cv2.CAP_PROP_FPS)

        # calculate duration of the video
        seconds = round(num_frames / video_fps, 2)
        video_time = datetime.timedelta(seconds=seconds)
        print(f"duration in seconds: {seconds}")
        print(f"video time: {video_time}")
        return num_frames


    @staticmethod
    def read_video(video_file, start_idx, end_idx):
        """Reads a video file, returns frames(T,H,W,3) """
        VidObj = cv2.VideoCapture(video_file)
        VidObj.set(cv2.CAP_PROP_POS_FRAMES, start_idx)
        # VidObj.set(cv2.CAP_PROP_POS_MSEC, 0)
        success, frame = VidObj.read()
        num_frames_to_read = end_idx - start_idx
        frames = list()
        frame_cnt = 0
        while success and frame_cnt <= num_frames_to_read:
            frame = cv2.cvtColor(np.array(frame), cv2.COLOR_BGR2RGB)
            frame = np.asarray(frame)
            frames.append(frame)
            frame_cnt += 1
            success, frame = VidObj.read()
        print(f"[DATA-LOADING] Read frames {start_idx}-{end_idx}")
        return np.asarray(frames)

    @staticmethod
    # TODO: convert function with pandas to read GT CSV and Extract BVPs
    def read_wave(nivs_gt_df, start_idx, end_idx):
        """Reads a bvp signal file."""
        # Add +1 as iLoc Indexing takes indexes form start_idx to end_idx-1
        bvp = nivs_gt_df["PPG"].iloc[start_idx:end_idx+1].values
        bvp = [float(x) for x in bvp]
        return np.asarray(bvp)


