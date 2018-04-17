# -*- coding: utf-8 -*-
import numpy as np
import pylab as pl
import cv2

sampling_rate = 8000
# N点FFT能精确计算的频率
# 假设取样频率为fs, 取波形中的N个数据进行FFT变换。
# 那么这N点数据包含整数个周期的波形时，FFT所计算的结果是精确的。
# 于是能精确计算的波形的周期是: n*fs/N。
# 对于8kHz取样，512点FFT来说，
# 8000/512.0 = 15.625Hz，前面的156.25Hz和234.375Hz正好是其10倍和15倍。
fft_size = 1024
t = np.arange(0, 1.0, 1.0 / sampling_rate)
x = np.sin(2 * np.pi * 156.25 * t) + 2 * np.sin(2 * np.pi * 234.375 * t)
xs = x[:fft_size]
xf = np.fft.rfft(xs) / fft_size
freqs = np.linspace(0, sampling_rate / 2, fft_size / 2 + 1)

# xfp = 20*np.log10/(np.clip(np.abs(xf), 1e-20, 1e100))
xfp = [abs(x) for x in xf]

pl.figure(figsize=(8, 4))
pl.subplot(211)
pl.plot(t[:fft_size], xs)
pl.xlabel(u"时间(秒)")
pl.title(u"156.25Hz和234.375Hz的波形和频谱")
pl.subplot(212)
pl.plot(freqs, xfp)
pl.xlabel(u"频率(Hz)")
pl.subplots_adjust(hspace=0.4)
pl.show()
