import matplotlib.pylab as plt
#for creating static, animated, and interactive visualizations in Python.
import cv2
import numpy as np #used to perform a wide variety of mathematical operations on arrays.
#supplies an enormous library of high-level mathematical functions that operate on these arrays and matrices.
image = cv2.imread('C:\\Users\\Admin\\PycharmProjects\\Lane_Detection\\road.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#cvtColor() method is used to convert an image from one color space to another.
# When converted to RGB, it will be saved as a correct image even if
# it is saved after being converted to a PIL(python imaging library ;adds support for opening, manipulating, and saving
# many different image file formats.

def region_of_interest(img, vertices):#Define a blank matrix that matches the image height/width.
    mask = np.zeros_like(img)
    #The zeros_like() function returns an array with element values as zeros.
    #The mask() method replaces the values of the rows where the condition evaluates to True.
    channel_count = img.shape[2]
    match_mask_color = (255,) * channel_count #channel count comes from this index;
    # As the pixel values range from 0 to 256,
    # apart from 0 the range is 255.
    cv2.fillPoly(mask, vertices, match_mask_color)
    #we return the image only where the mask pixel matches
    masked_image = cv2.bitwise_and(img, mask)
    #The bitwise_and operator returns an array that corresponds
    #to the resulting image from the merger of the given two images.
    return masked_image

print(image.shape)
height = image.shape[0]
width = image.shape[1]

region_of_interest_vertices = [
    (0, height),
    (width/2, height/2),
    (width, height)
]

cropped_image = region_of_interest(image,
                np.array([region_of_interest_vertices], np.int32),)

gray_image=cv2.cvtColor(cropped_image,cv2.COLOR_RGB2GRAY)
canny_image=cv2.Canny(gray_image, 100, 200)  #used to detect a wide range of edges in images


plt.imshow(cropped_image)
plt.show()
