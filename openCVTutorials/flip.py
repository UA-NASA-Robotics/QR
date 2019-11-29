# flipping image?
import cv2 

# Get original image from local folder
originalImage = cv2.imread('1.jpg')
# Flip the images
flipVertical = cv2.flip(originalImage, 0)
flipHorizontal = cv2.flip(originalImage, 1)
flipBoth = cv2.flip(originalImage, -1)

# Resize image
originalImage = cv2.resize(originalImage, (540, 960))
flipVertical = cv2.resize(flipVertical, (540,960))
flipHorizontal = cv2.resize(flipHorizontal, (540,960))
flipBoth = cv2.resize(flipBoth, (540,960))
            
cv2.imshow('Original image', originalImage)
cv2.imshow('Flipped vertical image', flipVertical)
cv2.imshow('Flipped horizontal image', flipHorizontal)
cv2.imshow('Flipped both image', flipBoth)

cv2.waitKey(0)
cv2.destroyAllWindows()

