import cv2
import numpy as np
import imutils
import pytesseract
import argparse
import re
import time

parser = argparse.ArgumentParser()
parser.add_argument('--i', '-image', help="Input image path", type= str)
parser.add_argument('--v', '-video', help="Input video path", type= str)

args = parser.parse_args()

def detected(img):
     saida = ''
     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
     bfilter = cv2.bilateralFilter(gray, 11, 17, 17) #Noise reduction
     edged = cv2.Canny(bfilter, 30, 200) #Edge detection
     keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
     contours = imutils.grab_contours(keypoints)
     contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
     location = None
     for contour in contours:
          approx = cv2.approxPolyDP(contour, 10, True)
          if len(approx) == 4:
               location = approx
               break
     if location is not None:          
          mask = np.zeros(gray.shape, np.uint8)

          cv2.drawContours(mask, [location], 0,255, -1)
          cv2.bitwise_and(img, img, mask=mask)
          executar = True
          try:
               (x,y) = np.where(mask==255)
               (x1, y1) = (np.min(x), np.min(y))
               (x2, y2) = (np.max(x), np.max(y))
               cropped_image = gray[x1:x2+1, y1:y2+1]
          except ValueError:  #raised if `y` is empty.
               executar = False 
               pass
      
          if executar:
               try:
                    image = cv2.resize(cropped_image, None, fx = 4, fy = 4,  interpolation = cv2.INTER_CUBIC)
                    image = cv2.GaussianBlur(image, (5, 5), 0)
                    image = cv2.threshold(image, 100, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY)[1]   
                    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
                    config = r'-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 --psm 6'
                    saida = pytesseract.image_to_string(image, lang='eng', config=config)
               except:
                    pass
     return saida

plate = ''
if args.i:
    img = cv2.imread(args.i)
    plate = detected(img)
elif args.v:
     frame_rate = 10
     prev = 0
     video = cv2.VideoCapture(args.v)
     if video.isOpened() == False:
        print("Não foi possivel abrir o video, verifique o arquivo informado")
     while video.isOpened():
          ret, frame = video.read()
          if (ret == False):
               break
          time_elapsed = time.time() - prev
          if time_elapsed > 2./frame_rate:
               prev = time.time()
               startTime = time.time() # reset time
               img = frame
               plate = detected(img)
               if len(plate) == 7:
                    match = re.match(r"([a-z]+)([0-9]+)", plate, re.I)
                    if match:
                         items = match.groups()
                         letras = items[0]
                         numeros = items[1]
                         if (len(letras) == 3 or len(letras) == 4) and (len(numeros) == 3 or len(numeros) == 4):
                              break

print(plate)