from pathlib import Path
import os

import numpy as np
import math

from torch import nn
from torch.optim import Adam
from torch.utils.data import Dataset
from torchvision import transforms as T, utils
from torchvision.transforms import InterpolationMode

import librosa
import audio_metadata

def read_audio_section(filename, start_sec: float, dur_sec: float):
    track = sf.SoundFile(filename)

    can_seek = track.seekable() # True
    if not can_seek:
        raise ValueError("Not compatible with seeking")

    sr = track.samplerate
    start_frame = sr * start_sec
    frame_len = sr * dur_sec
    track.seek(start_frame)
    audio_section = track.read(frame_len)
    return audio_section, sr

class ChunkedAudioDataset(Dataset):
    def __init__(
        self,
        folder,
        audio_len,
        samplerate = 8000,
        exts = ['mp3', 'wav'],
        start_silence_sec = 0
    ):
        super().__init__()
        self.folder = folder
        self.audio_len = audio_len
        self.samplerate = samplerate
        self.dur_sec = self.audio_len / samplerate
        self.start_silence_sec = start_silence_sec
        
        self.paths = [p for ext in exts for p in Path(f'{folder}').glob(f'*.{ext}')]

        durations_sec = np.array([audio_metadata.load(f).streaminfo['duration'] - start_silence_sec for f in self.paths])
        assert (durations_sec > 0).all(), "there is an audio shorter than {} sec".format(start_silence_sec)
        durations_frame = (durations_sec * samplerate).astype(int)
        num_chunks_per_file = durations_frame // audio_len - 1
        self.num_chunks_cumsum = np.cumsum(num_chunks_per_file)
        self.num_chunks_cumsum = np.insert(self.num_chunks_cumsum, 0, 0)
        self.num_chunks = self.num_chunks_cumsum[-1]

    def __len__(self):
        return self.num_chunks

    def find_file_id(self, index):
        file_id_low = 0
        file_id_high = len(self.num_chunks_cumsum)
        if file_id_high == 1:
            return 0
        else:
            while file_id_low < file_id_high:
                file_id_mid = math.floor((file_id_low + file_id_high)/2)
                
                this_chunk_id = self.num_chunks_cumsum[file_id_mid]
                next_chunk_id = self.num_chunks_cumsum[file_id_mid+1]
                if this_chunk_id <= index and next_chunk_id > index:
                    return file_id_mid
                elif this_chunk_id > index:
                    file_id_high = file_id_mid
                elif next_chunk_id <= index:
                    file_id_low = file_id_mid
                else:
                    assert 0, "invalid cumsum array"

    def __getitem__(self, index):
        file_id = self.find_file_id(index)
        audio_path = self.paths[file_id]
        index_start = self.num_chunks_cumsum[file_id]
        relative_index = index - index_start
        assert relative_index >= 0, "invalid find file id function"

        start_sec = relative_index * self.audio_len / self.samplerate
        dur_sec = self.audio_len / self.samplerate
        out_frame = librosa.load(audio_path, sr=self.samplerate, offset=start_sec+self.start_silence_sec, duration=self.dur_sec)[0]
        
        return out_frame
