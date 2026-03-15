from os.path import dirname, join, basename, isfile
from tqdm import tqdm

from models import SyncNet_color as SyncNet
import audio
from tensorboardX import SummaryWriter

import torch
from torch import nn
from torch import optim
import torch.backends.cudnn as cudnn
from torch.utils import data as data_utils
import numpy as np

from glob import glob

import os, random, cv2, argparse
from hparams import hparams, get_image_list
if True:
    filelist = glob(os.path.join("D:\\wav2lip_hd\\preprocessed_root\\data\\*\\", '*.wav'))

    random.shuffle(filelist)
    shuliang=0
    for vfile in tqdm(filelist):
        print(vfile)
        vidname=vfile[:-9]
        try:
                print(vidname)
                orig_mel_path = join(vidname, "audio.npy")
                if os.path.isfile(orig_mel_path):
                    #orig_mel = np.load(orig_mel_path)
                    print("在")
                else:
                    shuliang=shuliang+1
                    wavpath = join(vidname, "audio.wav")
                    wav = audio.load_wav(wavpath, hparams.sample_rate)

                    orig_mel = audio.melspectrogram(wav).T
                    np.save(orig_mel_path, orig_mel)
        except Exception as e:
                continue
    print(shuliang)
