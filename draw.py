import json
import matplotlib.pyplot as plt
import numpy as np

output_path = './output.json'
with open(output_path, 'r') as f:
    data = json.load(f)

ds_name = 'UVG'
video_name = 'Bosphorus_1920x1080_120'
rate_index = '000'

frame_bpp = data[ds_name][video_name][rate_index]['frame_bpp']
frame_psnr = data[ds_name][video_name][rate_index]['frame_psnr']

frame_bpp = np.array(frame_bpp)
frame_psnr = np.array(frame_psnr)

sort_idx = np.argsort(frame_bpp)

frame_bpp = frame_bpp[sort_idx]
frame_psnr = frame_psnr[sort_idx]

plt.figure(figsize=(10, 6))
plt.plot(frame_bpp, frame_psnr, marker='o', linestyle='-')
plt.title('BPP - PSNR Curve')
plt.xlabel('BPP')
plt.ylabel('PSNR')
plt.grid(True)
plt.show()