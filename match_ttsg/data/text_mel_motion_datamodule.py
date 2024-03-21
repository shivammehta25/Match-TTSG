import random
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import pyworld as pw
import torch
import torch.nn.functional as F
import torchaudio as ta
from lightning import LightningDataModule
from scipy.interpolate import interp1d
from torch.utils.data.dataloader import DataLoader

from match_ttsg.text import text_to_sequence
from match_ttsg.utils.audio import mel_spectrogram
from match_ttsg.utils.model import fix_len_compatibility, normalize
from match_ttsg.utils.pylogger import get_pylogger
from match_ttsg.utils.utils import (intersperse, to_torch,
                                    trim_or_pad_to_target_length)

log = get_pylogger(__name__)


def parse_filelist(filelist_path, split_char="|"):
    with open(filelist_path, encoding="utf-8") as f:
        filepaths_and_text = [line.strip().split(split_char) for line in f]
    return filepaths_and_text
 
    

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
        use_provided_durations,
        seed,
        generate_properties,
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
            self.hparams.use_provided_durations,
            self.hparams.seed,
            self.hparams.generate_properties,
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
            self.hparams.use_provided_durations,
            self.hparams.seed,
            self.hparams.generate_properties,
        )
        

    def train_dataloader(self):
        return DataLoader(
            dataset=self.trainset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
            collate_fn=TextMelMotionBatchCollate(self.hparams.n_spks, self.hparams.use_provided_durations)
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.validset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            collate_fn=TextMelMotionBatchCollate(self.hparams.n_spks, self.hparams.use_provided_durations)
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
    def __init__(
        self,
        name,
        filelist_path,
        n_spks,
        cleaners,
        motion_folder,
        add_blank=True,
        n_fft=1024,
        n_mels=80,
        sample_rate=22050,
        hop_length=256,
        win_length=1024,
        f_min=0.0,
        f_max=8000,
        data_statistics=None,
        use_provided_durations=False,
        seed=None,
        generate_properties=False,
    ):
        self.name = name
        self.filepaths_and_text = parse_filelist(filelist_path)
        self.n_spks = n_spks
        self.filelist_path = Path(filelist_path)
        self.motion_folder = Path(motion_folder)
        self.cleaners = cleaners
        self.add_blank = add_blank
        self.n_fft = n_fft
        self.n_mels = n_mels
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.win_length = win_length
        self.f_min = f_min
        self.f_max = f_max
        self.use_provided_durations = use_provided_durations
        self.generate_properties = generate_properties
        self.processed_folder_path = self.motion_folder.parent 

        if data_statistics is not None:
            self.data_statistics = data_statistics
        else:
            self.data_statistics = {
                'pitch_mean': 0,
                'pitch_std': 1,
                'energy_mean': 0,
                'energy_std': 1,
                'mel_mean': 0,
                'mel_std': 1,
                'motion_mean': 0,
                'motion_std': 1,
                'pitch_min': None,
                'pitch_max': None,
                'energy_min': None,
                'energy_max': None,
            }
        random.seed(seed)
        random.shuffle(self.filepaths_and_text)
        


    def get_data(self, filepath_and_text):
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
        
        if self.generate_properties:
            mel, energy = self.get_mel(filepath)
            pitch = self.get_pitch(filepath, mel.shape[1])
        else: 
            mel = np.load(self.processed_folder_path / 'mel' / Path(Path(filepath).stem).with_suffix(".npy"))
            mel = normalize(mel, self.data_statistics['mel_mean'], self.data_statistics['mel_std'])
            pitch = np.load(self.processed_folder_path / 'pitch' / Path(Path(filepath).stem).with_suffix(".npy"))
            pitch = normalize(pitch, self.data_statistics['pitch_mean'], self.data_statistics['pitch_std'])
            energy = np.load(self.processed_folder_path / 'energy' / Path(Path(filepath).stem).with_suffix(".npy"))
            energy = normalize(energy, self.data_statistics['energy_mean'], self.data_statistics['energy_std'])

        motion = self.get_motion(filepath, mel.shape[1])
        
        if self.use_provided_durations:
            durations = np.load(self.processed_folder_path / 'durations' / Path(Path(filepath).stem).with_suffix(".npy"))
            durations = torch.from_numpy(durations)
        else:
            durations = None
        
        return {
            'y': mel,
            'x': processed_text,
            'y_motion': motion,
            "pitch": pitch,
            "energy": energy,
            "durations": durations,
            'text': text,
            'spk': spk,
            'filepath': filepath
        }

    
    def get_motion(self, filename, mel_shape, ext=".expmap_86.1328125fps.pkl"):
        file_loc = self.motion_folder / Path(Path(filename).name).with_suffix(ext)
        motion = torch.from_numpy(pd.read_pickle(file_loc).to_numpy())
        motion = F.interpolate(motion.T.unsqueeze(0), mel_shape).squeeze(0)
        motion = normalize(motion, self.data_statistics['motion_mean'], self.data_statistics['motion_std'])
        return motion 

    def get_mel(self, filepath):
        audio, sr = ta.load(filepath)
        assert sr == self.sample_rate
        mel, energy = mel_spectrogram(audio, self.n_fft, self.n_mels, self.sample_rate, self.hop_length,
                              self.win_length, self.f_min, self.f_max, center=False)
        
        return mel.squeeze(0), energy.squeeze(0)
    
    def get_pitch(self, filepath, mel_length):
        _waveform, _sr = ta.load(filepath)
        _waveform = _waveform.squeeze(0).double().numpy() 
        assert _sr == self.sample_rate, f"Sample rate mismatch => Found: {_sr} != {self.sample_rate} = Expected"
        
        pitch, t = pw.dio(
            _waveform, self.sample_rate, frame_period=self.hop_length / self.sample_rate * 1000
        )
        pitch = pw.stonemask(_waveform, pitch, t, self.sample_rate)
        # A cool function taken from fairseq 
        # https://github.com/facebookresearch/fairseq/blob/3f0f20f2d12403629224347664b3e75c13b2c8e0/examples/speech_synthesis/data_utils.py#L99
        pitch = trim_or_pad_to_target_length(pitch,mel_length) 
        
        # Interpolate to cover the unvoiced segments as well 
        nonzero_ids = np.where(pitch != 0)[0]

        interp_fn = interp1d(
                nonzero_ids,
                pitch[nonzero_ids],
                fill_value=(pitch[nonzero_ids[0]], pitch[nonzero_ids[-1]]),
                bounds_error=False,
            )
        pitch = interp_fn(np.arange(0, len(pitch)))
        
        return pitch

    def get_text(self, text, add_blank=True):
        text_norm = text_to_sequence(text, self.cleaners)
        if self.add_blank:
            text_norm = intersperse(text_norm, 0) 
        text_norm = torch.IntTensor(text_norm)
        return text_norm

    def __getitem__(self, index):
        return self.get_data(self.filepaths_and_text[index])

    def __len__(self):
        return len(self.filepaths_and_text)

    def sample_test_batch(self, size):
        idx = np.random.choice(range(len(self)), size=size, replace=False)
        test_batch = []
        for index in idx:
            test_batch.append(self.__getitem__(index))
        return test_batch


class TextMelMotionBatchCollate:
    def __init__(self, n_spks, use_provided_durations=False) -> None:
        self.n_spks = n_spks
        self.use_provided_durations = use_provided_durations
        
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
        texts, filepaths = [], []
        spks = []
        pitches = torch.zeros((B, y_max_length), dtype=torch.float32)
        energies = torch.zeros((B, y_max_length), dtype=torch.float32)
        if self.use_provided_durations:
            durations = torch.zeros((B, 1, x_max_length), dtype=torch.float32)
        else:
            durations = None

        for i, item in enumerate(batch):
            y_, x_, y_motion_ = item['y'], item['x'], item['y_motion']
            y_lengths.append(y_.shape[-1])
            x_lengths.append(x_.shape[-1])
            y[i, :, :y_.shape[-1]] = to_torch(y_, torch.float)
            x[i, :x_.shape[-1]] = x_
            y_motion[i, :, :y_motion_.shape[-1]] = y_motion_
            pitches[i, :y_.shape[-1]] = to_torch(item['pitch'], torch.float)
            energies[i, :y_.shape[-1]] = to_torch(item['energy'], torch.float)
            texts.append(item['text'])
            filepaths.append(item['filepath'])
            spks.append(item["spk"])

            if self.use_provided_durations:
                durations[i, 0, :x_.shape[-1]] = item['durations']

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
            "spks": spks,
            'pitches': pitches,
            'energies': energies,
            'durations': durations,
            'filepaths': filepaths
        }