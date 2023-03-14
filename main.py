import cv2
import numpy as np
import os

threshold = 0.8  
folder_path = 'img'
files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
num_files = len(files)
num_files = num_files // 2
imgs = []
templates = []
show_res = True
AllBlack = False
imgGood = 0
imgBad = 0
for i in range(num_files):
    img = cv2.imread(f'img/{i}img.png', 0)
    template = cv2.imread(f'img/{i}obj.png', 0)
    imgs.append(img)
    templates.append(template)


def image_checker(img_c, template_c, i):
    global imgBad
    global imgGood
    h, w = template_c.shape[:2]
    res = cv2.matchTemplate(img_c, template_c, cv2.TM_CCOEFF_NORMED)
    loc = np.where(res >= threshold)
    mask = np.zeros_like(img_c)
    if loc[0].size > 0:
        for pt in zip(*loc[::-1]):
            if AllBlack:
                cv2.rectangle(mask, pt, (pt[0] + w, pt[1] + h), (255, 255, 255), -1)
                alpha = 0.4
                result = np.zeros_like(img_c)
                result[mask == 255] = img_c[mask == 255]
                if show_res:
                    image_new = cv2.addWeighted(img_c, alpha, result, 1 - alpha, 0)
                    cv2.imshow('Result', image_new)
                    cv2.waitKey(0)
                cv2.imwrite(f'result/{i}result.png', image_new)
            else:
                cv2.rectangle(img_c, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)
                if show_res:
                    cv2.imshow('Result', img_c)
                    cv2.waitKey(0)
                cv2.imwrite(f'result/{i}result.png', img_c)
        imgGood += 1
    else:
        imgBad += 1


def start_check():
    for i in range(num_files):
        image_checker(imgs[i], templates[i], i)
    else:
        print(f'На {imgGood} изображении/ях нашлось сходства.\nНа {imgBad} изображении/ях не нашлось сходство.')


start_check()
