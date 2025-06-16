import cv2
import numpy as np
import os

# 文件名列表
img_names = [
    "depth_color.png",
    "disparity_blended.png",
    "feature_matches_color.png",
    "snapshot01.png"
]

# 读取图片
imgs = []
for name in img_names:
    path = os.path.join(os.path.dirname(__file__), name)
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {name}")
    imgs.append(img)

# 统一尺寸
h, w = imgs[0].shape[:2]
imgs = [cv2.resize(img, (w, h)) for img in imgs]

# 拼接成2x2网格
row1 = np.hstack((imgs[0], imgs[1]))
row2 = np.hstack((imgs[2], imgs[3]))
grid = np.vstack((row1, row2))

# 保存
out_path = os.path.join(os.path.dirname(__file__), "summary_grid.png")
cv2.imwrite(out_path, grid)
print(f"Saved grid image to {out_path}")