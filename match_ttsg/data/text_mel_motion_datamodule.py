import random
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchaudio as ta
from einops import rearrange
from lightning import LightningDataModule
from torch.utils.data.dataloader import DataLoader
from tqdm.auto import tqdm

from match_ttsg.data.text_mel_datamodule import parse_filelist
from match_ttsg.text import text_to_sequence
from match_ttsg.text.symbols import symbols
from match_ttsg.utils.audio import mel_spectrogram
from match_ttsg.utils.model import fix_len_compatibility, normalize
from match_ttsg.utils.pylogger import get_pylogger
from match_ttsg.utils.utils import intersperse

log = get_pylogger(__name__)

        
    

class TextMelMotionDataModule(LightningDataModule):

    def __init__(
        self,
        name,
        train_filelist_path,
        valid_filelist_path,
        batch_size,
        num_workers,
        n_spks,
        pin_memory,
        cleaners,
        motion_folder,
        add_blank,
        n_fft,
        n_feats,
        n_motions,
        sample_rate,
        hop_length,
        win_length,
        f_min,
        f_max,
        data_statistics,
        motion_pipeline_filename,
        seed
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)
        
    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning with both `trainer.fit()` and `trainer.test()`, so be
        careful not to execute things like random split twice!
        """
        # load and split datasets only if not loaded already
        
        self.trainset = TextMelMotionDataset(
            self.hparams.name,
            self.hparams.train_filelist_path,
            self.hparams.n_spks,
            self.hparams.cleaners,
            self.hparams.motion_folder,
            self.hparams.add_blank,
            self.hparams.n_fft,
            self.hparams.n_feats,
            self.hparams.sample_rate,
            self.hparams.hop_length,
            self.hparams.win_length, 
            self.hparams.f_min,
            self.hparams.f_max,
            self.hparams.data_statistics,
            self.hparams.seed,
        )
        self.validset = TextMelMotionDataset(
            self.hparams.name,
            self.hparams.valid_filelist_path,
            self.hparams.n_spks,
            self.hparams.cleaners, 
            self.hparams.motion_folder,
            self.hparams.add_blank,
            self.hparams.n_fft,
            self.hparams.n_feats,
            self.hparams.sample_rate,
            self.hparams.hop_length,
            self.hparams.win_length,
            self.hparams.f_min,
            self.hparams.f_max,
            self.hparams.data_statistics,
            self.hparams.seed,
        )
        

    def train_dataloader(self):
        return DataLoader(
            dataset=self.trainset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
            collate_fn=TextMelMotionBatchCollate(self.hparams.n_spks)
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.validset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            collate_fn=TextMelMotionBatchCollate(self.hparams.n_spks)
        )

    def teardown(self, stage: Optional[str] = None):
        """Clean up after fit or test."""
        pass

    def state_dict(self):
        """Extra things to save to checkpoint."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Things to do when loading checkpoint."""
        pass


class TextMelMotionDataset(torch.utils.data.Dataset):
    def __init__(self, name, filelist_path, n_spks, cleaners, motion_folder, add_blank=True,
                 n_fft=1024, n_mels=80, sample_rate=22050,
                 hop_length=256, win_length=1024, f_min=0., f_max=8000, data_parameters=None, seed=None):
        self.name = name
        self.filepaths_and_text = parse_filelist(filelist_path)
        self.n_spks = n_spks
        self.filelist_path = Path(filelist_path)
        self.motion_fileloc = Path(motion_folder)        
        self.cleaners = cleaners
        self.add_blank = add_blank
        self.n_fft = n_fft
        self.n_mels = n_mels
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.win_length = win_length
        self.f_min = f_min
        self.f_max = f_max
        
        if data_parameters is not None:
            self.data_parameters = data_parameters
        else:
            self.data_parameters = { 'mel_mean': 0, 'mel_std': 1, 'motion_mean': 0, 'motion_std': 1 }
        random.seed(seed)
        random.shuffle(self.filepaths_and_text)


    def get_pair(self, filepath_and_text):
        if self.n_spks > 1:
            filepath, spk, text = (
                filepath_and_text[0],
                int(filepath_and_text[1]),
                filepath_and_text[2],
            )
        else:
            filepath, text = filepath_and_text[0], filepath_and_text[1]
            spk = None

        processed_text = self.get_text(text, add_blank=self.add_blank)
        mel = self.get_mel(filepath)
        motion = self.get_motion(filepath, mel.shape[1])
        
        return {
            'y': mel,
            'x': processed_text,
            'y_motion': motion,
            'text': text,
            'spk': spk
        }

    
    def get_motion(self, filename, mel_shape, ext=".expmap_86.1328125fps.pkl"):
        file_loc = self.motion_fileloc / Path(Path(filename).name).with_suffix(ext)
        motion = torch.from_numpy(pd.read_pickle(file_loc).to_numpy())
        motion = F.interpolate(motion.T.unsqueeze(0), mel_shape).squeeze(0)
        motion = normalize(motion, self.data_parameters['motion_mean'], self.data_parameters['motion_std'])
        return motion 

    def get_mel(self, filepath):
        audio, sr = ta.load(filepath)
        assert sr == self.sample_rate
        mel = mel_spectrogram(audio, self.n_fft, 80, self.sample_rate, self.hop_length,
                              self.win_length, self.f_min, self.f_max, center=False).squeeze()
        mel = normalize(mel, self.data_parameters['mel_mean'], self.data_parameters['mel_std'])
        return mel

    def get_text(self, text, add_blank=True):
        text_norm = text_to_sequence(text, self.cleaners)
        if self.add_blank:
            text_norm = intersperse(text_norm, 0) 
        text_norm = torch.IntTensor(text_norm)
        return text_norm

    def __getitem__(self, index):
        return self.get_pair(self.filepaths_and_text[index])

    def __len__(self):
        return len(self.filepaths_and_text)

    def sample_test_batch(self, size):
        idx = np.random.choice(range(len(self)), size=size, replace=False)
        test_batch = []
        for index in idx:
            test_batch.append(self.__getitem__(index))
        return test_batch


class TextMelMotionBatchCollate:
    def __init__(self, n_spks) -> None:
        self.n_spks = n_spks
        
    def __call__(self, batch):
        B = len(batch)
        y_max_length = max([item['y'].shape[-1] for item in batch])
        y_max_length = fix_len_compatibility(y_max_length)
        x_max_length = max([item['x'].shape[-1] for item in batch])
        n_feats = batch[0]['y'].shape[-2]
        n_motion = batch[0]['y_motion'].shape[-2]
        
        y = torch.zeros((B, n_feats, y_max_length), dtype=torch.float32)
        x = torch.zeros((B, x_max_length), dtype=torch.long)
        y_motion = torch.zeros((B, n_motion, y_max_length), dtype=torch.float32)
        y_lengths, x_lengths = [], []
        texts, spks = [], []

        for i, item in enumerate(batch):
            y_, x_, y_motion_ = item['y'], item['x'], item['y_motion']
            y_lengths.append(y_.shape[-1])
            x_lengths.append(x_.shape[-1])
            y[i, :, :y_.shape[-1]] = y_
            x[i, :x_.shape[-1]] = x_
            y_motion[i, :, :y_motion_.shape[-1]] = y_motion_
            texts.append(item['text'])
            spks.append(item["spk"])

        y_lengths = torch.LongTensor(y_lengths)
        x_lengths = torch.LongTensor(x_lengths)
        spks = torch.tensor(spks, dtype=torch.long) if self.n_spks > 1 else None

        return {
            'x': x, 
            'x_lengths': x_lengths, 
            'y': y, 
            'y_lengths': y_lengths, 
            'y_motion': y_motion, 
            'texts': texts,
            "spks": spks
        }
    
    
    
class TextMelMotionData2VecBatchCollate:
    def __init__(self, data2vec_hp) -> None:
        self.modalities = set(modality for modality in data2vec_hp if data2vec_hp[modality])

    
    def __call__(self, batch):
        B = len(batch)
        y_max_length = max([item['y'].shape[-1] for item in batch])
        y_max_length = fix_len_compatibility(y_max_length)
        x_max_length = max([item['x'].shape[-1] for item in batch])
        n_feats = batch[0]['y'].shape[-2]
        n_motion = batch[0]['y_motion'].shape[-2]
        max_data_2_vec_text = max([item['data2vec']['text'].shape[-1] for item in batch])
        
        if batch[0]['data2vec'] is not None:
                conditioning_tensor = torch.zeros((B, batch[0]['data2vec']['text'].shape[0], max_data_2_vec_text))
                conditioning_lengths = []
        else:
            conditioning_tensor = None
            conditioning_lengths = None

        y = torch.zeros((B, n_feats, y_max_length), dtype=torch.float32)
        x = torch.zeros((B, x_max_length), dtype=torch.long)
        
        y_motion = torch.zeros((B, n_motion, y_max_length), dtype=torch.float32)
        y_lengths, x_lengths = [], []
        texts, tokenized_texts = [], []

        for i, item in enumerate(batch):
            y_, x_, y_motion_ = item['y'], item['x'], item['y_motion']
            y_lengths.append(y_.shape[-1])
            x_lengths.append(x_.shape[-1])
            y[i, :, :y_.shape[-1]] = y_
            x[i, :x_.shape[-1]] = x_
            y_motion[i, :, :y_motion_.shape[-1]] = y_motion_
            texts.append(item['text'])
            
            if batch[0]['data2vec'] is not None:
                conditioning_lengths.append(item['data2vec']['text'].shape[-1])
                conditioning_tensor[i, :, :conditioning_lengths[-1]] = item['data2vec']['text']
                tokenized_texts.append(item['data2vec_tokenized_text'])

        y_lengths = torch.LongTensor(y_lengths)
        x_lengths = torch.LongTensor(x_lengths)
        
        if conditioning_lengths is not None:
            conditioning_lengths = torch.LongTensor(conditioning_lengths)        
        
        return {
            'x': x,
            'x_lengths': x_lengths,
            'y': y,
            'y_lengths': y_lengths,
            'y_motion': y_motion,
            'cond': conditioning_tensor,
            'cond_lengths': conditioning_lengths,
            'texts': texts,
            'tokenized_texts': tokenized_texts
        }