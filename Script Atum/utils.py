import cv2
import numpy as np

def getLAB(src):

    lab = cv2.cvtColor(src, cv2.COLOR_RGB2LAB)

    lLab = lab[:, :, 0].mean()
    aLab = lab[:, :, 1].mean()
    bLab = lab[:, :, 2].mean()

    return lLab, aLab, bLab

def getHSV(src):

    #alterando espaço de cores para HSV e jogando a nova imagem na variável hsv
    hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)

    return hsv

def getContours(hsv):

    #detectando a cor do sashibo por range de cor HSV (vermelho mais escuro até o vermelho mais claro)
    #Obs: Foram realizados vários testes com ranges diferentes de vermelho para se chegar nesses valores abaixo

    #definindo os valores HSV mínimo para detecção de cor
    lower_range = np.array([160, 0, 0])
    #definindo os valores HSV máximos para detecção de cor
    upper_range = np.array([180, 255, 255])

    #criando uma máscara na ára onde não for encontrada a cor que estiver no range definido acima
    mask = cv2.inRange(hsv, lower_range, upper_range)

    #imprimindo a máscara
    #cv2.imshow('mask', mask)

    #foi passado um filtro gausiano para suavizar as bordas da área do sashibo para minimizar as perdas
    gray = cv2.GaussianBlur(mask, (7, 7), 3)

    #aqui convertemos a imagem para bits, onde de o valor do pixel for menos que o limite
    # ele se torna 0 e se for maior se torna 1. Isso ajuda no algoritmo de contornos.
    t, dst = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_TRIANGLE) #reduzindo ruído da imagem utilizando o método threshold
    
    #contornando a área do sashibo
    contours, a = cv2.findContours(dst, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    return contours

def getLAB(src):

    lab = cv2.cvtColor(src, cv2.COLOR_RGB2LAB)

    #valuesLab = [L, A, B]

    return lab

def median(lst):

    sortedLst = sorted(lst)
    lstLen = len(lst)
    index = (lstLen - 1) // 2
   
    if (lstLen % 2):
        return sortedLst[index]
    else:
        return (sortedLst[index] + sortedLst[index + 1]) / 2.0
