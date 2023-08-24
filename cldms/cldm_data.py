from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from jukebox.make_models import make_vqvae, MODELS
from einops import repeat
import math
from jukebox.utils.dist_utils import print_all
from jukebox.data.labels import Labeller
from jukebox.utils.io import get_duration_sec, load_audio
import librosa
import numpy as np
import math
import jukebox.utils.dist_adapter as dist
from hparams import setup_hparams
from jbdiff.utils import device

def make_jb(train_data, control_data, level, batch_size, base_tokens, context_mult, aug_shift, num_workers, train=True):
    '''
    Constructs vqvae model as well as the dataloader for batching

    Params
    _______
    train_data: (str) location of audio files to train on
    level: (int) level of vqvae latent codes being trained on
    batch_size: (int) number of examples per batch for training
    base_tokens: (int) length of token sequence for diffusion on each level, multiply by 'level_mult' to translate from token length to sample length 
    aug_shift: (bool) if True, adds random cropping to each training example, if false, sequentially cuts up audio data
    num_workers: (int) number of workers for the dataloader, depends on training machine

    Returns
    _______
    Tuple of
    vqvae: (Jukebox) instance of the vqvae encoder/decoder for retrieving latent codes
    dataloader: (DataLoader) instance of DataLoader for loading training examples from directory of audio files
    hps: (dict) dictionary of hyperparameters
    '''
    base_model = "5b"    
    level_mult = 8 if level == 0 else 32 if level == 1 else 128
    sample_length = base_tokens*level_mult
    vqvae, *priors = MODELS[base_model]
    hps = setup_hparams(vqvae, dict(sample_length=sample_length, audio_files_dir=train_data, control_files_dir=control_data,labels=False, train_test_split=0.8, aug_shift=aug_shift, bs=batch_size))
    if train:
        dataset = Control_FilesAudioDataset(hps, context_mult)
        dataloader = DataLoader(dataset, batch_size=hps.bs, num_workers=num_workers, pin_memory=False, drop_last=True)
    else:
        dataloader = None
    vqvae = make_vqvae(hps, device)
    return vqvae, dataloader, hps

class Control_FilesAudioDataset(Dataset):
    '''
    Lifted from OpenAI Jukebox Repo, altered to return context as well as training batch

    Params
    ______

    - hps: hyperparameters built using setup_hyparams from jukebox repo

    '''
    def __init__(self, hps, context_mult):
        super().__init__()
        self.sr = hps.sr
        self.channels = hps.channels
        self.min_duration = hps.min_duration or math.ceil(hps.sample_length / hps.sr)
        self.max_duration = hps.max_duration or math.inf
        self.sample_length = hps.sample_length
        assert hps.sample_length / hps.sr < self.min_duration, f'Sample length {hps.sample_length} per sr {hps.sr} ({hps.sample_length / hps.sr:.2f}) should be shorter than min duration {self.min_duration}'
        self.aug_shift = hps.aug_shift
        self.labels = hps.labels
        self.init_dataset(hps)
        self.context_mult = context_mult

    def filter(self, files, control_files, durations, control_durations):
        # Remove files too short or too long
        keep = []
        control_keep = []
        for i in range(len(files)):
            if durations[i] / self.sr < self.min_duration:
                continue
            if durations[i] / self.sr >= self.max_duration:
                continue
            keep.append(i)
        for i in range(len(control_files)):
            if control_durations[i] / self.sr < self.min_duration:
                continue
            if control_durations[i] / self.sr >= self.max_duration:
                continue
            control_keep.append(i)
        print_all(f'self.sr={self.sr}, min: {self.min_duration}, max: {self.max_duration}')
        print_all(f"Keeping {len(keep)} of {len(files)} files")
        self.files = [files[i] for i in keep]
        self.control_files = [control_files[i] for i in control_keep]
        self.durations = [int(durations[i]) for i in keep]
        self.cumsum = np.cumsum(self.durations)

    def init_dataset(self, hps):
        # Load list of files and starts/durations
        files = librosa.util.find_files(f'{hps.audio_files_dir}', ['mp3', 'opus', 'm4a', 'aac', 'wav'])
        control_files = librosa.util.find_files(f'{hps.control_files_dir}', ['mp3', 'opus', 'm4a', 'aac', 'wav'])
        print_all(f"Found {len(files)} files. Getting durations")
        cache = dist.get_rank() % 8 == 0 if dist.is_available() else True
        durations = np.array([get_duration_sec(file, cache=cache) * self.sr for file in files])  # Could be approximate
        control_durations = np.array([get_duration_sec(file, cache=cache) * self.sr for file in control_files])
        self.filter(files, control_files, durations, control_durations)

        if self.labels:
            self.labeller = Labeller(hps.max_bow_genre_size, hps.n_tokens, self.sample_length, v3=hps.labels_v3)

    def get_index_offset(self, item):
        # For a given dataset item and shift, return song index and offset within song
        half_interval = self.sample_length//2
        shift = np.random.randint(-half_interval, half_interval) if self.aug_shift else 0
        offset = item * self.sample_length + shift # Note we centred shifts, so adding now
        midpoint = offset + half_interval
        assert 0 <= midpoint < self.cumsum[-1], f'Midpoint {midpoint} of item beyond total length {self.cumsum[-1]}'
        index = np.searchsorted(self.cumsum, midpoint)  # index <-> midpoint of interval lies in this song
        start, end = self.cumsum[index - 1] if index > 0 else 0.0, self.cumsum[index] # start and end of current song
        assert start <= midpoint <= end, f"Midpoint {midpoint} not inside interval [{start}, {end}] for index {index}"
        if offset > end - self.sample_length: # Going over song
            offset = max(start, offset - half_interval)  # Now should fit
        elif offset < start: # Going under song
            offset = min(end - self.sample_length, offset + half_interval)  # Now should fit
        assert start <= offset <= end - self.sample_length, f"Offset {offset} not in [{start}, {end - self.sample_length}]. End: {end}, SL: {self.sample_length}, Index: {index}"
        offset = offset - start
        context_offset = max(0, offset - self.sample_length*self.context_mult)
        return index, offset, context_offset

    def get_metadata(self, filename, test):
        """
        Insert metadata loading code for your dataset here.
        If artist/genre labels are different from provided artist/genre lists,
        update labeller accordingly.

        Returns:
            (artist, genre, full_lyrics) of type (str, str, str). For
            example, ("unknown", "classical", "") could be a metadata for a
            piano piece.
        """
        return None, None, None

    def get_song_chunk(self, index, offset, test=False):
        filename, total_length = self.files[index], self.durations[index]
        data, sr = load_audio(filename, sr=self.sr, offset=offset, duration=self.sample_length)
        assert data.shape == (self.channels, self.sample_length), f'Expected {(self.channels, self.sample_length)}, got {data.shape}'
        if self.labels:
            artist, genre, lyrics = self.get_metadata(filename, test)
            labels = self.labeller.get_label(artist, genre, lyrics, total_length, offset)
            return data.T, labels['y']
        else:
            return data.T
    
    def get_song_context_chunk(self, index, offset, context_offset, test=False):
        filename, control_filename, total_length = self.files[index], self.control_files[index], self.durations[index]
        data, sr = load_audio(filename, sr=self.sr, offset=offset, duration=self.sample_length)
        control_data, sr = load_audio(control_filename, sr=self.sr, offset=offset, duration=self.sample_length)
        context = repeat(np.zeros(data.shape), 'c t -> c (repeat t)', repeat = self.context_mult)
        context_data, _ = load_audio(filename, sr=self.sr, offset=context_offset, duration = self.sample_length*self.context_mult)
        length = int(offset - context_offset)
        context_data = context_data[:, :length]
        if length > 0:
            context[:, -length:] += context_data
        assert data.shape == (self.channels, self.sample_length), f'Expected {(self.channels, self.sample_length)}, got {data.shape}'
        if self.labels:
            artist, genre, lyrics = self.get_metadata(filename, test)
            labels = self.labeller.get_label(artist, genre, lyrics, total_length, offset)
            return data.T, labels['y'], control_data.T
        else:
            return data.T, context.T, control_data.T

    def get_item(self, item, test=False):
        index, offset, context_offset = self.get_index_offset(item)
        return self.get_song_context_chunk(index, offset, context_offset, test)

    def __len__(self):
        return int(np.floor(self.cumsum[-1] / self.sample_length))

    def __getitem__(self, item):
        return self.get_item(item)