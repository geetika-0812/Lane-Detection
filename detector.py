import matplotlib.pylab as plt
import cv2
import numpy as np

def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    #channel_count = img.shape[2]
    match_mask_color = 255
    cv2.fillPoly(mask, vertices, match_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def drow_the_lines(img, lines):
    img = np.copy(img)
    blank_image = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    #no. of channels for a colored image is always three

    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(blank_image,(x1,y1),(x2,y2),(0,255,0),thickness=2)
    img=cv2.addWeighted(img, 1.0, blank_image, 1, 0.0)
    #we are merging the blank image with the original image
    return img

# image = cv2.imread('road.jpg')
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
def process (image) :
    print(image.shape)
    height = image.shape[0]
    width = image.shape[1]

    region_of_interest_vertices = [
    (0, height),
    (width/2, height/2),
    (width, height)
    ]

    gray_image=cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    canny_image=cv2.Canny(gray_image, 100, 200)
    cropped_image = region_of_interest(canny_image,
                np.array([region_of_interest_vertices], np.int32),)

    lines=cv2.HoughLinesP(cropped_image,
                          rho=6,
                          theta=np.pi/180,
                          threshold=160,
                          lines=np.array([]),
                          minLineLength=40,
                          maxLineGap=25)
    image_with_lines = drow_the_lines(image,lines)
    return image_with_lines
# plt.imshow(cropped_image)
# plt.show()
cap=cv2.VideoCapture('C:\\Users\\Admin\\PycharmProjects\\Lane_Detection\\test.mp4')

while cap.isOpened():
    ret, frame = cap.read()
    frame = process(frame)
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows(1)


# video loop
# frame size
# perfect detection of green lines 