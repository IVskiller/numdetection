import numpy as np
import cv2
from sklearn.neighbors import KNeighborsClassifier

#######   training part    ###############
samples = np.loadtxt('generalsamples.data', np.float32)

responses = np.loadtxt('generalresponses.data', np.float32)

responses = responses.reshape((responses.size, 1))



model = KNeighborsClassifier(n_neighbors=3)
model.fit(samples, responses)

############################# testing part  #########################


im = cv2.imread('f.png')







out = np.zeros(im.shape, np.uint8)
gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
thresh = cv2.adaptiveThreshold(gray, 255, 1, 1, 11, 2)

contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

for i, cnt in enumerate(contours):
    if cv2.contourArea(cnt) > 50:
        [x, y, w, h] = cv2.boundingRect(cnt)
        if h > 28:
            skip = False
            for j in range(len(contours)):
                if i != j:
                    [x_other, y_other, w_other, h_other] = cv2.boundingRect(contours[j])
                    if x >= x_other and y >= y_other and x + w <= x_other + w_other and y + h <= y_other + h_other:
                        skip = True
                        break
            if not skip:
                cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 2)
                roi = thresh[y:y + h, x:x + w]
                roismall = cv2.resize(roi, (10, 10))
                roismall = roismall.reshape((1, 100))
                roismall = np.float32(roismall)
                results = model.predict(roismall)
                string = str(int(results[0]))
                print(string)
                cv2.putText(out, string, (x, y + h), 0, 1, (0, 255, 0))

cv2.namedWindow("in", cv2.WINDOW_NORMAL)
cv2.namedWindow("out", cv2.WINDOW_NORMAL)
cv2.imshow('in', im)
cv2.imshow('out', out)
cv2.waitKey(0)
