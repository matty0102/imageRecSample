# -*- coding:utf-8 -*-

import sys
import numpy as np
from skimage import io
from skimage import transform
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def compute_score_map(template, target):
	# 拡大縮小しない場合と同様にscore_mapを計算
	th, tw = template.shape[0:2]
	# 各位置におけるSSDを保存する変数
	score_map = np.zeros(shape = (target.shape[0] - th,
								  target.shape[1] - tw))
	# 画像全体を走査して、SSDを計算
	for y in range(score_map.shape[0]):
		print("DEBUG:calc_line: %d" % y)
		for x in range(score_map.shape[1]):
			diff = target[y:y + th, x:x + tw] - template
			score_map[y, x] = np.square(diff).sum()
	return score_map

def main():
	# コマンドライン引数受け取り
	template_path = sys.argv[1]
	target_path = sys.argv[2]

	# テンプレート、対象画像の読み込み（グレースケール）
	template = io.imread(template_path, as_gray = True)
	target = io.imread(target_path, as_gray = True)

	# 画像を2^1/8ずつ縮小しながら各スケールのscore_mapを計算
	score_map = []
	scale_factor = 2.0 ** (-1.0 / 8.0)
	target_scaled  = target + 0
	for s in range(8):
		score_map.append(compute_score_map(template, target_scaled))
		target_scaled = transform.rescale(target_scaled, scale_factor)

	# SSDが最小のスケール・座標を取得
	score, s, (x, y) = min([(np.min(score_map), 
					 	     s,
						     np.unravel_index(np.argmin(score_map), score_map.shape))
						     for s, score_map in enumerate(score_map)]
						   )

	# 結果を可視化
	fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(8,3))
	ax1.imshow(template, cmap=cm.Greys_r)
	ax1.set_axis_off()
	ax1.set_title('template')
	# targetの上にマッチした領域を矩形で囲う
	ax2.imshow(target, cmap=cm.Greys_r)
	ax2.set_axis_off()
	scale = (scale_factor ** s)
	ax2.set_title('target')
	th, tw = template.shape[0:2]
	rect = plt.Rectangle((y/scale, x/scale), tw, th, 
						  edgecolor='r', facecolor='none')
	print("x: {0}, y: {1}, scale: {2:0<5.3}".format(x, y, scale))
	print("xp: {0}, yp: {1}".format(x/scale, y/scale))
	ax2.add_patch(rect)
	plt.show()

#ax3.imshow(score_map, cmap=cm.Greys_r)
#ax3.set_axis_off()
#ax3.set_title('score_map')
## マッチした領域を矩形で囲う
#ax3.add_patch(plt.Rectangle((y - th/2, x - tw/2), tw, th, edgecolor='w', facecolor='none', linewidth=2.5))
#plt.show()

if __name__ == "__main__":
	main()










