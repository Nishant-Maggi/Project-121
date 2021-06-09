import cv2
import time
import numpy as np 

four_cc = cv2.VideoWriter_fourcc(*"XVID")
out_file = cv2.VideoWriter("output.avi", four_cc, 20.0, (640, 480))
capture = cv2.VideoCapture(0)

time.sleep(2)

bg = 0

for i in range(0, 60):
    ret, bg = capture.read()

bg = np.flip(bg, axis = 1)

while(capture.isOpened()):
    ret, img = capture.read()

    if not ret:
        break

    img = np.flip(img, axis = 1)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower_red = np.array([0, 120, 50])
    upper_red = np.array([10, 255, 255])

    mask1 = cv2.inRange(hsv, lower_red, upper_red)
    
    lower_red = np.array([170, 120, 70])
    upper_red = np.array([180, 255, 255])

    mask2 = cv2.inRange(hsv, lower_red, upper_red)

    mask_add = mask1 + mask2
    mask_add = cv2.morphologyEx(mask_add, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask_add = cv2.morphologyEx(mask_add, cv2.MORPH_DILATE, np.ones((3, 3), np.uint8))

    finished_mask = cv2.bitwise_not(mask_add)

    res1 = cv2.bitwise_and(img, img, mask = finished_mask)
    res2 = cv2.bitwise_and(bg, bg, mask = mask_add)

    final_out = cv2.addWeighted(res1, 1, res2, 1, 0)
    out_file.write(final_out)

    cv2.imshow("Cloak", final_out)
    cv2.waitKey(1)
    
capture.release()
out_file.release()
cv2.destroyAllWindows()