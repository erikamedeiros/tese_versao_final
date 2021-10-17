import cv2
import numpy as np
import sys
import os
import argparse
import convert
import math
import csv
import statistics as statistic
import utils
from matplotlib import pyplot as plt

#função responsável por extrair as informações do sashibo
def test(location):

    #abrindo imagem
    src = cv2.imread(location)
    #redimensionando a imagem para 600x400
    src = cv2.resize(src, (600, 400))

    lab = utils.getLAB(src)

    lLab = lab[:, :, 0].mean()
    aLab = lab[:, :, 1].mean()
    bLab = lab[:, :, 2].mean()

    #alterando espaço de cores para HSV e jogando a nova imagem na variável hsv
    hsv = utils.getHSV(src)

    contours = utils.getContours(hsv)

    #pintando contornos na imagem para visualização
    #cv2.drawContours(src, contours, -1, (0, 0, 255), 1, cv2.LINE_AA)

    #variáveis responsáveis por armazenar os valores da posição do sashibo na imagem original
    x = 0
    y = 0
    w = 0
    h = 0

    #Percorrendo todo contorno para extrair as informações da posição do sashibo
    for c in contours:
        area = cv2.contourArea(c)
        if area > 1000 and area < 1000000:
            #criando um retângulo na área encontrada e passando as informações de ponto e área para as variáveis
            (x, y, w, h) = cv2.boundingRect(c)
            #cv2.rectangle(src, (x, y), (x + w, y + h), (0, 255, 0), 2, cv2.LINE_AA)
            #print(x,y,w,h)
            break
        else:
            """spl = location.split('/')
            spl = spl[len(spl) - 1]
            print('<<< ERRO AO PROCESSAR IMAGEM >>> ' + spl)
            print('')"""
            return -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0

    #extraindo sashibo
    crop = src[y:y+h, x:x+w]

    #abaixo só trabalhamos com a imagem que foi extraída (que é o sahsibo):

    hsv_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

    #padrão usado pelo opencv, vamos usar nas plotagens dos gráficos abaixo
    color = ('b','g','r')

    #plotando gráficos:

    plt.figure(figsize=(6, 4))

    plt.suptitle('Extração de informações do ' + location)

    grid = plt.GridSpec(2, 3, wspace=0.4, hspace=0.3)

    plt.subplot(grid[0, 0])
    plt.title('Sashibo')
    plt.imshow(crop[:,:,::-1])

    plt.subplot(grid[0, 1:])
    plt.title('Histograma referente')

    histr = list()

    #calculando histograma para o RGB
    for i, col in enumerate(color):
        histTemp = cv2.calcHist([crop], [i], None, [256], [0, 256])
        histr.append(histTemp)
        plt.plot(histTemp, color = col)
        plt.xlim([0, 256])
    
    #sortedList = histr.sort()

    #CALCULANDO MODA DO RGB NO HISTOGRAMA

    arrayMaximums = list() #TODO: colocar esse algoritmo em utils
    arrayMedians = list()

    histR = 120
    histG = 155
    histB = 162

    median = 0

    medianR = 0
    medianG = 0
    medianB = 0 

    for c in histr:
        temp = list()
        for i in c:
            for j in i:
                if j > 150: #se o valor for maior que 150 ele faz a adição da informação do histograma no array
                    temp.append(j)

        #print(sorted(temp, reverse=True))

        #temp = quantidade de pixels que estão acima de 150 no canal R, G ou B. Abaixo calculamos a média para temp.

        if len(temp) > 0:
            modeTemp = statistic.mean(temp) #tirando a média dos valores encontrados que são acima de 150.
            arrayMaximums.append(modeTemp) #calculando a moda e inserindo valor no array
            median = utils.median(temp)
            arrayMedians.append(median)
    
    if len(arrayMaximums) == 3:
        histR = arrayMaximums[2]
        histG = arrayMaximums[1]
        histB = arrayMaximums[0]
        medianR = arrayMedians[2]
        medianG = arrayMedians[1]
        medianB = arrayMedians[0]
    else:
        spl = location.split('/')
        spl = spl[len(spl) - 1]
        print('<<< ERRO AO PROCESSAR IMAGEM >>> ' + spl)

    #print(len(histr))

    #arrayMaximums é o array com a quantidade de pixels pra BGR que possuem as características BGR

    #print(arrayMaximums) # posicao 0 = B posicao 1 = G posicao 2 = R

    plt.subplot(grid[1, :2])
    plt.title('HSV')

    x = np.arange(3)

    #capturando dados HSV da imagem separados
    hue = hsv_crop[:, :, 0].mean()
    saturation = hsv_crop[:, :, 1].mean()
    valueHsv = hsv_crop[:, :, 2].mean()

    #RGB

    r = crop[:, :, 0].mean()
    g = crop[:, :, 1].mean()
    b = crop[:, :, 2].mean()

    values_RGB = [r, g, b]

    #print(values_RGB)

    values = [saturation, hue, valueHsv]

    plot = plt.bar(x, values)

    #plotando HSV em gráfico
    for value in plot:
        height = value.get_height()
        plt.text(value.get_x() + value.get_width()/2., 1.002*height,'%d' % int(height), ha='center', va='bottom')

    plt.xticks(x, ('Saturação', 'Matiz', 'Valor'))
    plt.xlabel("hsv separado")
    plt.ylabel("Valor") 

    #convertendo RGB para HSI
    hsi = convert.RGB_TO_HSI(crop)

    """r = crop[:, :, 0].mean()
    b = crop[:, :, 1].mean()
    b = crop[:, :, 2].mean()
    print('RGB')
    print()
    print(r)
    print(g)
    print(b)"""

    #cv2.imshow('HSI Image', hsi)

    #capturando dados HSI da imagem separados
    hueHsi = hsi[:, :, 0].mean()
    saturationHsi = hsi[:, :, 1].mean()
    intensity = hsi[:, :, 2].mean()

    if math.isnan(float(hueHsi)):
        hueHsi = 0

    values1 = [float(saturationHsi), float(hueHsi), float(intensity)]

    #print(values1)

    plt.subplot(grid[1, 2])
    plt.title('HSI')

    plt.xticks(x, ('Saturação', 'Matiz', 'Intensidade'))
    plt.xlabel("hsi separado")
    plt.ylabel("Valor") 

    x = np.arange(3)
    plot = plt.bar(x, values1)

    for value in plot:
        height = value.get_height()
        plt.text(value.get_x() + value.get_width()/2., 1.002*height,'%f' % float(height), ha='center', va='bottom')

    #cv2.imshow('contornos', src)

    #cv2.imshow('crop', crop)

    #print(hue)
    #print(saturation)

    plt.gcf().canvas.set_window_title(location)

    #exibindo graficos
    #plt.show(block=False)

    imgName = location.split('/')

    print('Processado: ' + imgName[len(imgName) - 1])

    return 0, imgName[len(imgName) - 1], medianR, medianG, medianB, histR, histG, histB, r, g, b, saturation, hue, valueHsv, saturationHsi, hueHsi, intensity, lLab, aLab, bLab

def listDir(arg):
    files = []
    for a, _, arq in os.walk(arg):
        if len(arq) != 0:
            for file in arq:
                files.append(a + "/" + file)
    return files
    

#função responsável por coletar os argumentos (nome dos aquivos de imagem) passados na chamada do programa 
def main(argv):

    counter = 0
    counterError = 0

    print('Processando imagens...')

    #print(argv[1])

    files = listDir(argv[1])

    print(files)

    with open('DATASET.csv', mode='w', newline='') as csv_file:
    
        fieldnames = ["imgName", "medianR", "medianG", "medianB", "histR", "histG", "histB", "r", "g", "b", "saturationHsv", "hueHsv", "valueHsv", "saturationHsi", "hueHsi", "intensityHsi", "lLab", "aLab", "bLab"]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames, delimiter=';')
        writer.writeheader()
        
        for n in files:
            cod, imgName, medianR, medianG, medianB, histR, histG, histB, r, g, b, saturation, hue, valueHsv, saturationHsi, hueHsi, intensity, lLab, aLab, bLab = test(n)
            if cod < 0:
                #print('Erro ao processar imagem')
                counterError = counterError + 1
                continue
            else:
                counter = counter + 1
                writer.writerow({"imgName": imgName, "medianR": medianR, "medianG": medianG, "medianB": medianB, "histR": histR, "histG": histG, "histB": histB, "r": r, "g": g, "b": b, "saturationHsv": float(saturation), "hueHsv": float(hue), 
                "valueHsv": float(valueHsv), "saturationHsi": float(saturationHsi), "hueHsi": float(hueHsi), "intensityHsi": float(intensity),
                "lLab": lLab, "aLab": aLab, "bLab": bLab})
        csv_file.close()

    print('<<< Processadas ' + str(counter) + ' imagens >>>')
    print('<<< Qtd de falhas => ' + str(counterError) + ' imagens >>>')

    #pausando aplicação para manter as janelas dos gráficos abertas
    while True:
        key = cv2.waitKey(1)
        #se clicar ESC o programa fecha (caso n feche é só encerrar o processo pelo terminal (Control + C))
        if key == 27:
            cv2.destroyAllWindows()
    return 0


#chamada da função main
if __name__ == '__main__':
    sys.exit(main(sys.argv))