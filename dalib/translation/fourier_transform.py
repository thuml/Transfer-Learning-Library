import torch
import numpy as np
import os
import json
import tqdm
import random
from PIL import Image


def low_freq_mutate_np(amp_src, amp_trg, b=0):
    a_src = np.fft.fftshift(amp_src, axes=(-2, -1))
    a_trg = np.fft.fftshift(amp_trg, axes=(-2, -1))

    _, h, w = a_src.shape
    # b = (np.floor(np.amin((h,w))*L)).astype(int)
    c_h = np.floor(h/2.0).astype(int)
    c_w = np.floor(w/2.0).astype(int)

    h1 = c_h-b
    h2 = c_h+b+1
    w1 = c_w-b
    w2 = c_w+b+1

    a_src[:,h1:h2,w1:w2] = a_trg[:,h1:h2,w1:w2]
    a_src = np.fft.ifftshift( a_src, axes=(-2, -1) )
    return a_src


def source_to_target( src_img, trg_img, beta=0):
    # exchange magnitude
    # input: src_img, trg_img

    src_img_np = src_img #.cpu().numpy()
    trg_img_np = trg_img #.cpu().numpy()

    # get fft of both source and target
    fft_src_np = np.fft.fft2( src_img_np, axes=(-2, -1) )
    fft_trg_np = np.fft.fft2( trg_img_np, axes=(-2, -1) )

    # extract amplitude and phase of both ffts
    amp_src, pha_src = np.abs(fft_src_np), np.angle(fft_src_np)
    amp_trg, pha_trg = np.abs(fft_trg_np), np.angle(fft_trg_np)

    # mutate the amplitude part of source with target
    amp_src_ = low_freq_mutate_np( amp_src, amp_trg, b=beta)

    # mutated fft of source
    fft_src_ = amp_src_ * np.exp( 1j * pha_src )

    # get the mutated image
    src_in_trg = np.fft.ifft2( fft_src_, axes=(-2, -1) )
    src_in_trg = np.real(src_in_trg)

    return src_in_trg


class FourierTransform:
    def __init__(self, image_list, amplitude_dir, beta=0, rebuild=False):
        self.amplitude_dir = amplitude_dir
        if not os.path.exists(amplitude_dir) or rebuild:
            os.makedirs(amplitude_dir, exist_ok=True)
            self.build_amplitude(image_list, amplitude_dir)
        self.beta = beta
        self.length = len(image_list)

    @staticmethod
    def build_amplitude(image_list, amplitude_dir):
        # extract amplitudes from target domain
        for i, image_name in tqdm.tqdm(enumerate(image_list)):
            image = Image.open(image_name).convert('RGB')
            image = np.asarray(image, np.float32)
            image = image.transpose((2, 0, 1))
            fft = np.fft.fft2(image, axes=(-2, -1))
            amp = np.abs(fft)
            np.save(os.path.join(amplitude_dir, "{}.npy".format(i)), amp)

    def __call__(self, image):
        amp_trg = np.load(os.path.join(self.amplitude_dir, "{}.npy".format(random.randint(0, self.length-1))))

        image = np.asarray(image, np.float32)
        image = image.transpose((2, 0, 1))

        # get fft, amplitude on source domain
        fft_src = np.fft.fft2(image, axes=(-2, -1))
        amp_src, pha_src = np.abs(fft_src), np.angle(fft_src)
        # mutate the amplitude part of source with target
        amp_src_ = low_freq_mutate_np(amp_src, amp_trg, b=self.beta)

        # mutated fft of source
        fft_src_ = amp_src_ * np.exp(1j * pha_src)

        # get the mutated image
        src_in_trg = np.fft.ifft2(fft_src_, axes=(-2, -1))
        src_in_trg = np.real(src_in_trg)

        src_in_trg = src_in_trg.transpose((1, 2, 0))
        src_in_trg = Image.fromarray(src_in_trg.clip(min=0, max=255).astype('uint8')).convert('RGB')

        return src_in_trg
