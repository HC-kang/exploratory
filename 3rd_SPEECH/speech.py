# import numpy as np
# import matplotlib.pyplot as plt
# def single_tone(frequency, sampling_rate = 16000, duration = 1):
#     t = np.linspace(0, duration, int(sampling_rate))
#     y = np.sin(2*np.pi*frequency*t)
#     return y
# y = single_tone(400)

# plt.plot(y[:41])
# plt.show()

# plt.stem(y[:41])
# plt.show()

###########
import numpy as np
import os

data_path = '/Users/heechankang/projects/pythonworkspace/git_study/speech_wav_8000.npz'
speech_data = np.load(data_path)
print('완료')

print('Wave data shape : ', speech_data['wav_vals'].shape)
print('Label data shape : ', speech_data['label_vals'].shape)
print('완료')

import IPython.display as ipd
import random

# 데이터 선택
rand = random.randint(0, len(speech_data['wav_vals']))
print('rand num : ', rand)

sr = 8000 # 1초당 재생되는 샘플의 갯수
data = speech_data['wav_vals'][rand]
print('Wave data shape : ', data.shape)
print('label : ', speech_data['label_vals'][rand])

ipd.Audio(data, rate=sr)