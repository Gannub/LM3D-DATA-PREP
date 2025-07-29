import numpy as np
import os

# path = "/mnt/nas/sitt/demo/SIGGRAPH2025_DEMO/GaussianAvatars/media/306/flame_param.npz"
path = "/mnt/nas/jea/VHAP/export/monocular/baka_whiteBg_staticOffset_maskBelowLine/flame_param/new_flame.npz"

data = np.load(path)

print(data.files)

print(data['dynamic_offset'].dtype)
# arr = np.array([0.0, 0.0, 0.0])

# np.savez("lubta.npz", **{key: data[key] if key != 'translation' else arr for key in data.files})

