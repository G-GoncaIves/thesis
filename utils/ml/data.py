import torch
from torch.utils.data import Dataset
import numpy as np
import os
import pandas as pd

class Videos(Dataset):
    
    _tuples_label_list = ["im_ra", "im_dec", "im_td", "im_mag", "im_kappa", "im_tdd"]

    def __init__(
        self,
        videos_dir,
        generation_df_path,
        labels,
        rescale,
        no_td = True
    ):
        self.no_td = no_td
        total_df = pd.read_pickle(generation_df_path)
        _labels = [str(label) for label in labels]
        self.df = total_df[["id",*_labels]]
        self.labels = []
        self.standardize_df(labels=_labels)
        self.data_dir = videos_dir
        ids = os.listdir(videos_dir)
        self.ids = list(filter(self.is_desired_type, ids))
        self.rescale = rescale
    
    def is_desired_type(self, sample_name):
        if "no_td" in sample_name:
            _no_td = True 
        else:
            _no_td = False
        return _no_td == self.no_td
        
    def standardize_df(self, labels):
        for label in labels:
            if label in self._tuples_label_list:
                im0_label, im1_label = self.split_columns(current_column_name=label)
                self.labels.append(im0_label)
                self.labels.append(im1_label)
            else:
                self.labels.append(label)
                
    def split_columns(self, current_column_name):
        im0_label = current_column_name.replace("im","im0")
        im1_label = current_column_name.replace("im","im1")
        self.df[[im0_label, im1_label]] = pd.DataFrame(self.df[current_column_name].to_list(), index=self.df.index)
        return im0_label, im1_label
    
    def get_target_tensor(self, id):
        row = self.df.loc[self.df['id'] == id]
        target_tensor = torch.zeros(len(self.labels))
        for i in range(len(self.labels)):
            target_tensor[i] = float(row[self.labels[i]].to_numpy()[0])
        return target_tensor
    
    def get_sample_tensor(self, id):
        video_path = os.path.join(self.data_dir, id).replace("\\","/")
        video_array = np.float32(np.load(video_path))
        frame_array = np.asarray([video_array])
        return torch.from_numpy(frame_array)
        
    def __len__(self):
        return len(self.ids)
    
    def __getitem__(self, index):
        sample_id = self.ids[index]
        frame_tensor = self.get_sample_tensor(id=sample_id)
        if self.no_td:
            id_str_extras = "no_td.npy"
        else:
            id_str_extras = ".npy"
        target_tensor = self.get_target_tensor(id = sample_id.replace(id_str_extras,""))
        return frame_tensor, target_tensor*self.rescale, index
