# -*- coding:utf-8 -*-

import sys
import numpy as np
from skimage import io
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# (1) コマンドライン引数受け取り
template_path = sys.argv[1]
target_path = sys.argv[2]

# (2) テンプレート、対象画像の読み込み（グレースケール）
template = io.imread(template_path, as_gray = True)
target = io.imread(target_path, as_gray = True)
th, tw = template.shape[0:2]

# (3) 各位置におけるSSDを保存する変数
score_map = np.zeros(shape = (target.shape[0] - th,
							  target.shape[1] - tw))

# (4) 画像全体を走査して、SSDを計算
for y in range(score_map.shape[0]):
	for x in range(score_map.shape[1]):
		diff = target[y:y + th, x:x + tw] - template
		score_map[y, x] = np.square(diff).sum()

# (5) SSDが最小の座標を取得
x, y = np.unravel_index(np.argmin(score_map), score_map.shape)

# (6) 結果を可視化
fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(8,3))
ax1.imshow(template, cmap=cm.Greys_r)
ax1.set_axis_off()
ax1.set_title('template')

ax2.imshow(target, cmap=cm.Greys_r)
ax2.set_axis_off()
ax2.set_title('target')
# マッチした領域を矩形で囲う
ax2.add_patch(plt.Rectangle((y,x), tw, th, edgecolor='w', facecolor='none', linewidth=2.5))
#ax2.add_patch(rect)

ax3.imshow(score_map, cmap=cm.Greys_r)
ax3.set_axis_off()
ax3.set_title('score_map')
# マッチした領域を矩形で囲う
ax3.add_patch(plt.Rectangle((y - th/2, x - tw/2), tw, th, edgecolor='w', facecolor='none', linewidth=2.5))
plt.show()









