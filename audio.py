import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import math
from PIL import Image
import random
# import signal
import wave
import os
from scipy import signal
import imageio
import scipy
from skimage.exposure import exposure, equalize_hist
class WAVE(object):
    def __init__(self, path, showimg=True):
        wavfile = wave.open(path, "rb")
        params = wavfile.getparams()
        self.path = path
        self.nchannels, self.sampwidth, self.framerate, self.nframes = params[:4]
        strData = wavfile.readframes(self.nframes)  # 读取音频，字符串格式
        waveData = np.frombuffer(strData, dtype=np.int16)  # 将字符串转化为int
        wavfile.close()
        waveData = waveData * 1.0 / (max(abs(waveData)))  # wave幅值归一化
        self.waveData = np.reshape(waveData, [self.nframes, self.nchannels]).T[0]
        self.time = np.round(np.arange(0, self.nframes) * (1.0 / self.framerate), 3)
        if showimg:
            print("采样率", self.framerate)
            print("总采样点数", self.nframes)
            print("声道数", self.nchannels)
            print("audio_data", self.waveData)
            print("time", self.time)


def wavread(path):
    wavfile = wave.open(path, "rb")
    params = wavfile.getparams()
    framesra, frameswav = params[2], params[3]
    datawav = wavfile.readframes(frameswav)
    wavfile.close()
    datause = np.fromstring(datawav, dtype=np.short)
    datause.shape = -1, 2
    datause = datause.T
    time = np.arange(0, frameswav) * (1.0 / framesra)
    return datause, time


def enframe(signal, nw, inc):
    '''将音频信号转化为帧。
    参数含义：
    signal:原始音频型号
    nw:每一帧的长度(这里指采样点的长度，即采样频率乘以时间间隔)
    inc:相邻帧的间隔（同上定义）
    '''
    signal_length = len(signal)  # 信号总长度
    if signal_length <= nw:  # 若信号长度小于一个帧的长度，则帧数定义为1
        nf = 1
    else:  # 否则，计算帧的总长度
        nf = int(np.ceil((1.0 * signal_length - nw + inc) / inc))
    pad_length = int((nf - 1) * inc + nw)  # 所有帧加起来总的铺平后的长度
    zeros = np.zeros((pad_length - signal_length,))  # 不够的长度使用0填补，类似于FFT中的扩充数组操作
    pad_signal = np.concatenate((signal, zeros))  # 填补后的信号记为pad_signal
    indices = np.tile(np.arange(0, nw), (nf, 1)) + np.tile(np.arange(0, nf * inc, inc),
                                                           (nw, 1)).T  # 相当于对所有帧的时间点进行抽取，得到nf*nw长度的矩阵
    indices = np.array(indices, dtype=np.int32)  # 将indices转化为矩阵
    frames = pad_signal[indices]  # 得到帧信号
    #    win=np.tile(winfunc(nw),(nf,1))  #window窗函数，这里默认取1
    #    return frames*win   #返回帧信号矩阵
    return frames


def enframe(signal, length_perframe, gap_between_frame, winfunc):
    '''将音频信号转化为帧。
    参数含义：
    signal:原始音频型号
    length_perframe:每一帧的长度(这里指采样点的长度，即采样频率乘以时间间隔)
    gap_between_frame:相邻帧的间隔（同上定义）
    '''
    signal_length = len(signal)  # 信号总长度
    if signal_length <= length_perframe:  # 若信号长度小于一个帧的长度，则帧数定义为1
        nf = 1
    else:  # 否则，计算帧的总长度
        nf = int(np.ceil((1.0 * signal_length - length_perframe + gap_between_frame) / inc))
    pad_length = int((nf - 1) * gap_between_frame + length_perframe)  # 所有帧加起来总的铺平后的长度
    zeros = np.zeros((pad_length - signal_length,))  # 不够的长度使用0填补，类似于FFT中的扩充数组操作
    pad_signal = np.concatenate((signal, zeros))  # 填补后的信号记为pad_signal
    indices = np.tile(np.arange(0, length_perframe), (nf, 1)) + np.tile(
        np.arange(0, nf * gap_between_frame, gap_between_frame), (length_perframe, 1)).T  # 相当于对所有帧的时间点进行抽取，得到nf*nw长度的矩阵
    indices = np.array(indices, dtype=np.int32)  # 将indices转化为矩阵
    frames = pad_signal[indices]  # 得到帧信号
    win = np.tile(winfunc, (nf, 1))  # window窗函数，这里默认取1
    return frames * win  # 返回帧信号矩阵


def readFile(path):
    for root, dirs, files in os.walk(path):
        return files


def MatrixToImage(data):
    data = data * 255
    new_im = Image.fromarray(data.astype(np.uint8))
    return new_im


if __name__ == '__main__':
    path = "audio"

    fileNames = readFile(path)
    for file in fileNames:
        wav = WAVE(path + "/" + file, True)
        plt.plot(wav.time, wav.waveData)
        plt.show()
        sp_data, freqs, bins, im = plt.specgram(wav.waveData, Fs=512)
        sp_img = MatrixToImage(sp_data)
        sp_data = equalize_hist(sp_data)
        plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
        plt.tight_layout()
        plt.savefig("img")
        plt.axis("off")
        plt.show()
    exit(1)

    fft_size = 512
    inc = 128
    winfunc = signal.hamming(fft_size)
    data = np.array(wav.waveData[0])

    Frame = enframe(data, fft_size, inc, winfunc)
    Frame_ft = np.fft.rfft(Frame) / fft_size

    fig, (ax1, ax2) = plt.subplots(nrows=2)
    data, freqs, bins, im = ax1.specgram(data, Fs=512)
    ax1.axis('tight')

    # We need to explictly set the linear threshold in this case...
    # Ideally you should calculate this from your bin size...
    ax2.set_yscale('symlog', linthreshy=0.01)

    ax2.pcolormesh(bins, freqs, 100 * np.log10(data))
    ax2.axis('tight')

    plt.show()
    exit(1)
    plt.plot(Frame_ft)
    plt.show()
    plt.specgram(data, 2048, 1024)
    plt.show()
    plt.plot(wav.waveData[0])
    plt.show()
    # exit(1)
