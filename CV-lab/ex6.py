import cv2


img = cv2.imread("input.jpg")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))


eroded = cv2.erode(gray, kernel, iterations=2)


cv2.imshow("Original Image", gray)
cv2.imshow("Eroded Image", eroded)

cv2.waitKey(0)
cv2.destroyAllWindows()