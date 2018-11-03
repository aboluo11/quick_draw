from lightai.core import *
import ujson
import cv2

def draw_cv2(raw_strokes, size=256, lw=6):
    img = np.zeros((size, size), np.float32)
    for stroke in raw_strokes:
        for i in range(len(stroke[0]) - 1):
            _ = cv2.line(img, (stroke[0][i], stroke[1][i]), (stroke[0][i + 1], stroke[1][i + 1]), 255, lw)
    return img

def row2sample(row):
    x = ujson.loads(row['drawing'])
    y = row['y']
    return draw_cv2(x), y