import cv2
import numpy as np


def imread_objects_from_path(original_image_path, mask_path):
    img = cv2.imread(original_image_path)
    img_mask = cv2.imread(mask_path, 0) # the 0 should mean to read it in as black/white
    return img, img_mask


# this takes in the original image (imread object) and an alpha mask (probably from the Salient Object Dectecor)
# and finds the bounding rectangle for the object,
# and then writes to file a vew cropped image
# but it should probably just return the image? or take in another arg that seys where to write it to.
def get_bounding_rect_for_isolated_obj(original_image, gray, output_folder, img_id):

    #ret, threshed_img = cv2.threshold(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 127, 255, cv2.THRESH_BINARY)

    retval, thresh_gray = cv2.threshold(gray, thresh=100, maxval=255, \
                                       type=cv2.THRESH_BINARY_INV)

    contours, hierarchy = cv2.findContours(thresh_gray,cv2.RETR_LIST, \
                                       cv2.CHAIN_APPROX_SIMPLE)

    gray_size = gray.shape[0] * gray.shape[1]
    #print(gray_size)
    # Find object with the biggest bounding box
    mx = (0,0,0,0)      # biggest bounding box so far
    mx_area = 0
    for cont in contours:
        x,y,w,h = cv2.boundingRect(cont)
        area = w*h
        if area == gray_size:
            #do nothing - the bounding rect is the whole image - nobody wants that!
            pass
        elif area > mx_area:
            mx = x,y,w,h
            mx_area = area
    x,y,w,h = mx

    # Output to files
    roi=original_image[y:y+h,x:x+w]
    #print(gray.shape)
    #print(x,y,w,h)
    extracted_file_name = img_id + '_crop.jpg'
    #cv2.imwrite(output_folder + extracted_file_name, roi)

    cv2.imwrite(extracted_file_name, roi)


    cv2.rectangle(original_image,(x,y),(x+w,y+h),(200,0,0),2)
    #cv2.imwrite(output_folder + img_id + '_cont.jpg', original_image)
    return extracted_file_name


# get_bounding_rect_for_isolated_obj(img, img_mask)


def view_crop_main(orig_path, mask_path, output_folder, img_id):
    # img, img_mask = imread_objects_from_path(orig_path,mask_path)
    img = cv2.imread(orig_path)
    img_mask = cv2.imread(mask_path, 0)
    extracted_file_name = get_bounding_rect_for_isolated_obj(img, img_mask, output_folder, img_id)
    return extracted_file_name




def crop_rect(img_path, rect_coords):
    #print(rect_coords)
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    x = int(rect_coords['left'])
    y = int(rect_coords['top'])
    h = int(rect_coords['height'])
    w = int(rect_coords['width'])
    print(rect_coords)
    crop_img = img[y:y+h, x:x+w]
    cv2.imwrite(img_path, crop_img)

#resizes an image to be 300 x 300 maintaining ratio
def img_resize(img_name, img_id):
	img = cv2.imread(img_name,0)
	resized_file_name = img_id + "_resized.jpg"
	h, w = img.shape[:2]
	#print(h, w)

# https://www.pyimagesearch.com/2014/01/20/basic-image-manipulations-in-python-and-opencv-resizing-scaling-rotating-and-cropping/
	if(w>h): #horizontal
		#print("horizontal")
		#resize the width to be 300 pixels
		r = 300.0 / img.shape[1]
		dim = (300, int(img.shape[0] * r))
	 
		resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
		#cv2.imwrite("resized_" + img_name,resized)

		vertical_space =300 - resized.shape[0]

		#creates an array that is one side of padding that will be concatenated to cropped image 
		#on the top and bottom to create an image exactly 300 x 300 for the pixel comparison later
		blank_edge = np.zeros((vertical_space//2, 300), np.uint8)

		temp_array1 = np.concatenate((blank_edge, resized), axis=0)
		temp_array2 = np.concatenate((temp_array1, blank_edge), axis=0)


		#if the vertical space is odd then we need to add an extra pixel to the edge
		#to make it exactly 300 down
		if(vertical_space%2==0):
			cv2.imwrite(resized_file_name, temp_array2)
		else:
			pixel_line = np.zeros((1, 300), np.uint8)
			temp_array3 = np.concatenate((temp_array2, pixel_line), axis=0)
			cv2.imwrite(resized_file_name, temp_array3)


	else: #vertical
		#print("vertical")
		#resize the height to be 300 pixels
		r = 300.0 / img.shape[0]
		dim = (int(img.shape[1] * r), 300)
	 
		resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
		#cv2.imwrite("resized_" + img_name, resized) 

		horizontal_space =300 - resized.shape[1]

		#creates an array that is one side of padding that will be concatenated to cropped image 
		#on both sides to create an image exactly 300 x 300 for the pixel comparison later
		blank_edge = np.zeros((300, horizontal_space/2), np.uint8)

		temp_array1 = np.concatenate((blank_edge, resized), axis=1)
		temp_array2 = np.concatenate((temp_array1, blank_edge), axis=1)


		#if the horizontal space is odd then we need to add an extra pixel to the edge
		#to make it exactly 300 across
		if(horizontal_space%2==0):
			cv2.imwrite(resized_file_name, temp_array2)
		else:
			pixel_line = np.zeros((300, 1), np.uint8)
			temp_array3 = np.concatenate((temp_array2, pixel_line), axis=1)
			cv2.imwrite(resized_file_name,temp_array3)

	return resized_file_name

#input: two equal sized images to see how well they match
def match_shapes(img1, img2):
	match_sum = 0
	max_sum = img1.shape[0] * img1.shape[1] * 255
	#print("max_sum ", max_sum)
	for i, img1_px in np.ndenumerate(img1):
		img2_px = img2[i]
		match_sum = match_sum + abs(int(img1_px) - int(img2_px))

	proportion_correct = 1 - (match_sum/float(max_sum))
	return proportion_correct

def test_pair(img1_name, img1_id, img2_name, img2_id):
	img1 = cv2.imread(img1_name,0)

	#img2 = cv2.imread('oval_contour.png',0)
	#img2_id = 'oval'

	img2 = cv2.imread(img2_name,0)

	#img2 = cv2.imread('circle_contour2.png',0)
	#img2_id = 'circle'

	ret, thresh = cv2.threshold(img1, 127, 255,0)
	ret, thresh2 = cv2.threshold(img2, 127, 255,0)
	contours,hierarchy = cv2.findContours(thresh,2,1)

	#cv2.imshow('thresh',thresh)
	#cv2.waitKey(0)
	#cv2.destroyAllWindows()


	#cnt1 = contours[0]
	contours,hierarchy = cv2.findContours(thresh2,2,1)
	cnt2 = contours[0]

	#crop tree image
	cropped1 = get_bounding_rect_for_isolated_obj(img1, img1, 'OpenCV_Testing', img1_id)
	#print(cropped1)
	img3 = cv2.imread(cropped1,0)
	#cv2.imshow('cropped',img3)
	#cv2.waitKey(0)
	#cv2.destroyAllWindows()
	#print ret

	resized1 = img_resize(cropped1, img1_id)
	resized_img1 = cv2.imread(resized1,0)

	#crop triangle image
	cropped2 = get_bounding_rect_for_isolated_obj(img2, img2, 'OpenCV_Testing', img2_id)
	#print(cropped2)
	img3 = cv2.imread(cropped2,0)
	#cv2.imshow('cropped',img3)
	#cv2.waitKey(0)
	#cv2.destroyAllWindows()
	#print ret

	resized2 = img_resize(cropped2, img2_id)
	resized_img2 = cv2.imread(resized2,0)

	proportion = match_shapes(resized_img1, resized_img2)

	#match_test_img1 = np.zeros((2, 2), np.uint8)
	#match_test_img2 = np.zeros((2, 2), np.uint8)
	#match_test_img2 = np.full((2, 2),255, np.uint8)
	return img1_id + "+" + img2_id + "," + str(proportion) + ",\r,"


##########################################################
results = open("results.txt","a")
test_set = [("antlers.png","antlers"),("bauble.png","bauble"),("bell.png","bell"),("boot.png","boot"),
			("clover.png","clover"),("deer.png","deer"), ("flowers.png","flowers"),("gourd.png","gourd"),
			("lamb.png","lamb"),("palm.png","palm"), ("pine_cone.png","pine_cone"),("pine.png","pine"),
			("pizza.png","pizza"),("present.png","present"), ("shoe.png","shoe"),("shovel.png","shovel"),
			("sundae.png","sundae"),("sweater.png","sweater"),("tree.jpeg","tree"),("wreath.png","wreath")]

alphabet_set = [("A.png","A"),("a_lower.png","a_lower"),("B.png","B"),("D.png","D"),("I.png","I"),
				("i_lower.png","i_lower"),("L.png","L"),("n.png","n"),("O.png","O"),
				("o_lower.png","o_lower"),("p_lower.png","p_lower"),("P.png","P"),("T.png","T")]

for image in test_set:
	for letter in alphabet_set:
		print(image[0],image[1],letter[0],letter[1])
		results.write(test_pair(image[0],image[1],letter[0],letter[1]))





