import cv2 
import numpy as np 
import matplotlib.pyplot as plt 
import cvlib as cv 
from cvlib.object_detection import draw_bbox 
from numpy.lib.polynomial import poly

TF_ENABLE_ONEDNN_OPTS=0


imagem = cv2.imread('meteor_callenge_01')

imagem_rgb = cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(10,10))

plt.axis("off")

plt.imshow(imagem_rgb)