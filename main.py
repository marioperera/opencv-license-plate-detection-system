import numpy as np
import cv2
import  imutils
import sys
import pytesseract
import pandas as pd
import time




def extract_characters(img):
    bw_image = cv2.bitwise_not(img)
    contours = cv2.findContours(bw_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[1]

    char_mask = np.zeros_like(img)
    bounding_boxes = []

    small_count = 0
    medium_count = 0
    large_count = 0
    areas = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        areas.append(area)

        if (area > 1000) and (area < 100000):
            small_count = small_count + 1

        if (area > 2500) and (area < 100000):
            medium_count = medium_count + 1

        if (area > 5000) and (area < 100000):
            large_count = large_count + 1
    min_ar = 1000
    if large_count > 5:
        min_ar = 5000
    elif medium_count > 5:
        min_ar = 2500
    elif small_count > 5:
        min_ar = 1000
    else:
        # no number plate found
        return -1, -1
    # print(areas)

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        center = (x + w / 2, y + h / 2)
        if (area > min_ar) and (area < 100000):
            x, y, w, h = x - 4, y - 4, w + 8, h + 8
            bounding_boxes.append((center, (x, y, w, h)))
            cv2.rectangle(char_mask, (x, y), (x + w, y + h), 255, -1)

    cv2.imwrite('outputs/licence_plate_mask3.png', char_mask)

    clean = cv2.bitwise_not(cv2.bitwise_and(char_mask, char_mask, mask=bw_image))

    bounding_boxes = sorted(bounding_boxes, key=lambda item: item[0][0])

    characters = []
    for center, bbox in bounding_boxes:
        x, y, w, h = bbox
        char_image = clean[y:y + h, x:x + w]
        characters.append((bbox, char_image))

    print(characters)
    return clean, characters


def highlight_characters(img, chars):
    output_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for bbox, char_img in chars:
        x, y, w, h = bbox
        cv2.rectangle(output_img, (x, y), (x + w, y + h), 255, 1)

    return output_img



def getPlateNumber(img):
    # we need to keep in mind aspect ratio so the image does
    # not look skewed or distorted -- therefore, we calculate
    # the ratio of the new image to the old image
    r = 300.0 / img.shape[1]
    dim = (300, int(img.shape[0] * r))

    # perform the actual resizing of the image and show it
    img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

    # end resize

    clean_img, chars = extract_characters(img)

    # if not match found
    if type(clean_img) is int and clean_img == -1:
        return -1

    output_img = highlight_characters(clean_img, chars)
    cv2.imwrite('outputs/licence_plate_out.png', output_img)

    samples = np.loadtxt('data_generation/char_samples.data', np.float32)
    responses = np.loadtxt('data_generation/char_responses.data', np.float32)
    responses = responses.reshape((responses.size, 1))

    model = cv2.ml.KNearest_create()
    model.train(samples, cv2.ml.ROW_SAMPLE, responses)

    plate_chars = ""
    for bbox, char_img in chars:
        small_img = cv2.resize(char_img, (10, 10))
        small_img = small_img.reshape((1, 100))
        small_img = np.float32(small_img)
        retval, results, neigh_resp, dists = model.findNearest(small_img, k=1)
        plate_chars += str(chr((results[0][0])))

    return plate_chars



image = cv2.imread('7.jpg')

image = imutils.resize(image, width=500)

cv2.imshow("Original Image", image)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#cv2.imshow("1 - Grayscale Conversion", gray)

gray = cv2.bilateralFilter(gray, 18, 18, 17)
#cv2.imshow("2 - Bilateral Filter", gray)


gray = cv2.GaussianBlur(gray,(3,3), 0)
#comment for 1,6,7,20


edged = cv2.Canny(gray, 170, 200)
#cv2.imshow("4 - Canny Edges", edged)

(new, cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cnts=sorted(cnts, key = cv2.contourArea, reverse = True)[:30]
NumberPlateCnt = None

count = 0
for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            NumberPlateCnt = approx
            break

# Masking the part other than the number plate
mask = np.zeros(gray.shape,np.uint8)
new_image = cv2.drawContours(mask,[NumberPlateCnt],0,255,-1)
new_image = cv2.bitwise_and(image,image,mask=mask)
cv2.namedWindow("Final_image",cv2.WINDOW_NORMAL)
cv2.imshow("Final_image",new_image)
cv2.imwrite('test3.png',new_image)

testimg =cv2.imread('test3.png')

gray = cv2.cvtColor(testimg, cv2.COLOR_BGR2GRAY)
cv2.imshow('filtered image',gray)
cv2.imwrite('test3.png',gray)
getPlateNumber(gray)


# ============================================================================

# Configuration for tesseract
config = ('-l eng --oem 1 --psm 3')

# Run tesseract OCR on image
text = pytesseract.image_to_string(gray, config=config)
print(text)

#Data is stored in CSV file
raw_data = {'date': [time.asctime( time.localtime(time.time()) )],
        'v_number': [text]}

df = pd.DataFrame(raw_data, columns = ['date', 'v_number'])
df.to_csv('data.csv')

# Print recognized text
print(text)

cv2.waitKey(0)