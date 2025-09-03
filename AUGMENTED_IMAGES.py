import cv2 as cv
import numpy as np
import os


#Augmentation functions

def rotate(image, angle,rotPoint=None):
    #get image dimensions
    (height,width) = image.shape[:2]

    if rotPoint is None:
        rotPoint =(width//2,height//2)

    rotation_matrix = cv.getRotationMatrix2D(rotPoint,angle,scale=1.0)
    dimension=(width,height)
    return cv.warpAffine(image,rotation_matrix,dimension)

def translate(image,x,y):
    (height,width)=image.shape[:2]

    translation_matrix=np.float32([[1,0,x],[0,1,y]])
    dimension=(width,height)

    return cv.warpAffine(image,translation_matrix,dimension)

def flip(image,flip_code=1):
    #flip an image horizontally(1),vertically(0),both(-1)
     return cv.flip(image,flip_code)

def gaussian_blur(image,kernel_size=(5,5)):
    
    return cv.GaussianBlur(image,kernel_size,0) #0 indicates sigMax value to calualte the std deviation based on the kernel size to control the shape of the blur automatically.


 


#MAIN_SCRIPT





#define the directory to save augmented images
output_dir='output'

#Create the output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"Created directory: {output_dir}")

img = cv.imread('Photos/dog.jpg')

#get image dimensions
(height,width)=img.shape[:2]



#to check if image is loaded successfully

if img is None:
    print(f"Error: could n0t read the {img}. Make sure the file exists")
else:
    print("Image loaded successfully")
    rotated_image = rotate(img,15)
    translated_image = translate(img,75,75)
    flipped_image = flip(img,0)
    
    #to create multiple blurred images 
    blurred_image_list=[]  
    kernel_sizes_to_try = [(3,3),(5,5),(7,7),(9,9),(11,11),(15,15)]
    print("Generating blurred image of different kernels")

    for size in kernel_sizes_to_try:
        #creating an empty list to store the blurred images of different kernels
       
        blurred_image=gaussian_blur(img,kernel_size=size)
        filename = f"blurred_{size[0]}X{size[1]}.jpg"
        #to sve the images in different files
        cv.imwrite(os.path.join(output_dir,filename.jpg),blurred_image)
        print(f"generated filename: {filename}")
        blurred_image_list.append(blurred_image)
       
    

        
 


#save the results

cv.imwrite(os.path.join(output_dir,'rotated.jpg'),rotated_image)
cv.imwrite(os.path.join(output_dir,'translated.jpg'),translated_image)
cv.imwrite(os.path.join(output_dir,'flipped.jpg'),flipped_image)


print(f"Augmented images saved to '{output_dir}' using functions")

#display

cv.imshow('original',img)
cv.imshow('rotated',rotated_image)
cv.imshow('translated',translated_image)
cv.imshow('flipped',flipped_image)
# Enumerate over the list to get (index, image) pairs.
# i is the index position in the list (starting at 0)
# image is the image at that index (blurred_images_list[i])
# This creates a separate window for each image, labeled starting from 1.
for i, image in enumerate(blurred_image_list):
    window_name = f"Blurred Image {i+1}"
    cv.imshow(window_name, image)


cv.waitKey(0)
cv.destroyAllWindows()







