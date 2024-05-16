import cv2
import numpy as np
import matplotlib.pyplot as plt
import cvlib as cv
from cvlib.object_detection import draw_bbox
from numpy.lib.polynomial import poly


imagem = cv2.imread('/content/meteor_challenge_01_ajustada.png')

imagem_rgb = cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB)
######### MELHORAR A BASE DE DADOS
# Remover a o céu, a parte de baixo dos lagos e as letras, uma vez que não são necessários no problema
cor_ceu = np.array([2, 119, 189]) #Cor em RGB do céu
cor_letras = np.array([52, 134, 183])  #Cor em RGB da letra
tolerancia = 30

# Criar máscaras para o céu e as letras
mascara_ceu = np.linalg.norm(imagem_rgb - cor_ceu, axis=-1) < tolerancia
mascara_letras = np.linalg.norm(imagem_rgb - cor_letras, axis=-1) < tolerancia

# Trocando os pixels equivalentes ao céu e as letras por preto
imagem_rgb[mascara_ceu] = [0, 0, 0]
imagem_rgb[mascara_letras] = [0, 0, 0]

######### CONTAR OS OBJETOS
def contar_objetos(imagem, cor_rgb):
  cor_baixa = np.array(cor_rgb) - 30
  cor_alta = np.array(cor_rgb) + 30
  mascara = cv2.inRange(imagem, cor_baixa, cor_alta)
  num_objetos = cv2.countNonZero(mascara)
  return num_objetos

meteoros = contar_objetos(imagem_rgb, np.array([255, 255, 255]))
estrelas = contar_objetos(imagem_rgb, np.array([255, 0, 0]))
print(f'Número de meteoros: {meteoros}')
print(f'Número de estrelas: {estrelas}')

contador = 0
for i in range(1,6):
  imagem = cv2.imread(f'/content/imgCortadaMeteoro{i}.png')
  imagem_cortada = cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB)
  meteoros = contar_objetos(imagem_cortada, np.array([255, 255, 255]))
  contador += meteoros

print(f'O número de meteoros que cairá na água é igual a: {contador}')

