import cv2
import numpy as np
import os

output_folder = '/Users/samross/Desktop/Main/Barnard/Research/Shape_Matching/OpenCV_Testing/Main_Test_Set/contour_match/out/'

output_folder2 = '/Users/samross/Desktop/Main/Barnard/Research/Shape_Matching/OpenCV_Testing/Main_Test_Set/contour_match/out2/'
output_folder3 = '/Users/samross/Desktop/Main/Barnard/Research/Shape_Matching/OpenCV_Testing/Main_Test_Set/contour_match/out3/'

def get_bounding_rect_for_isolated_obj_contour(img, img_id):
  imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  ret, thresh = cv2.threshold(imgray, 127, 255, 0)
  contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

  img_size = img.shape[0] * img.shape[1]

  # Find object with the biggest bounding box
  mx = (0,0,0,0)      # biggest bounding box so far
  mx_area = 0
  winning_contour = None
  for cont in contours:
     x,y,w,h = cv2.boundingRect(cont)
     area = w*h
     if area == img_size:
         #do nothing - the bounding rect is the whole image - nobody wants that!
         pass
     elif area > mx_area:
         mx = (x,y,w,h)
         mx_area = area
         winning_contour = cont
  x,y,w,h = mx

  contour_shape = (h,w)
 #this is the part that makes the contour image, -1 = fill, 255 = white
  contour_mask = np.zeros(contour_shape, np.uint8)
  # contour_mask = np.zeros(gray.shape, np.uint8)
  cv2.drawContours(contour_mask, [winning_contour], -1, (255), -1, offset=(-x,-y))
  file_name = img_id + '_contour.jpg'
  cv2.imwrite(output_folder + file_name, contour_mask)

  return contour_mask

def img_resize(img):
	# resized_file_name = img_id + "_resized.jpg"
	h, w = img.shape[:2]

# https://www.pyimagesearch.com/2014/01/20/basic-image-manipulations-in-python-and-opencv-resizing-scaling-rotating-and-cropping/
	if(w>h): #horizontal
		#resize the width to be 300 pixels
		r = 300.0 / img.shape[1]
		dim = (300, int(img.shape[0] * r))
	 
		resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

		vertical_space =300 - resized.shape[0]

		#creates an array that is one side of padding that will be concatenated to cropped image 
		#on the top and bottom to create an image exactly 300 x 300 for the pixel comparison later
		blank_edge = np.zeros((vertical_space//2, 300), np.uint8)

		temp_array1 = np.concatenate((blank_edge, resized), axis=0)
		temp_array2 = np.concatenate((temp_array1, blank_edge), axis=0)

		#if the vertical space is odd then we need to add an extra pixel to the edge
		#to make it exactly 300 down
		if(vertical_space%2==0):
			return temp_array2
		else:
			pixel_line = np.zeros((1, 300), np.uint8)
			temp_array3 = np.concatenate((temp_array2, pixel_line), axis=0)
			return temp_array3

	else: #vertical
		#resize the height to be 300 pixels
		r = 300.0 / img.shape[0]
		dim = (int(img.shape[1] * r), 300)
	 
		resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

		horizontal_space =300 - resized.shape[1]

		#creates an array that is one side of padding that will be concatenated to cropped image 
		#on both sides to create an image exactly 300 x 300 for the pixel comparison later
		blank_edge = np.zeros((300, horizontal_space/2), np.uint8)

		temp_array1 = np.concatenate((blank_edge, resized), axis=1)
		temp_array2 = np.concatenate((temp_array1, blank_edge), axis=1)

		#if the horizontal space is odd then we need to add an extra pixel to the edge
		#to make it exactly 300 across
		if(horizontal_space%2==0):
			return temp_array2
		else:
			pixel_line = np.zeros((300, 1), np.uint8)
			temp_array3 = np.concatenate((temp_array2, pixel_line), axis=1)
			return temp_array3

def contour_match(img, img_id, letter, letter_id):
	#bitwise and the contour_shape
	bit_and = cv2.bitwise_and(img, letter)
	bit_or = cv2.bitwise_or(img, letter)

	name_and = img_id + '_' + letter_id + '_and.png'
	cv2.imwrite(output_folder2 + name_and, bit_and)

	name_or = img_id + '_' + letter_id + '_or.png'
	cv2.imwrite(output_folder3 + name_or, bit_or)

	#count the 1's in the result
	match_and = cv2.countNonZero(bit_and)
	match_or = cv2.countNonZero(bit_or)

	#calculate match using Jaccard Similarity
	if(float(match_or) != 0):
		similarity = float(match_and) / float(match_or)
		return similarity
	else:
		return -1 #The or of the two image is zero i.e. both solid black


######################## TEST FUNCTIONS #######################
def test_one():
	img_file = './test_img/bowtie_macaroni.jpg'
	img_id = 'bowtie_macaroni'
	results = []
	letter_path = './alphabet/'

	img = cv2.imread(img_file)
	img_contour = get_bounding_rect_for_isolated_obj_contour(img, img_id)
	img_resized = img_resize(img_contour)

	for cap_letter in os.listdir(letter_path):
		letter_id = cap_letter.replace(".png",'')
		letter = cv2.imread(letter_path + cap_letter, cv2.IMREAD_GRAYSCALE)
		similarity = contour_match(img_resized, img_id, letter, letter_id)
		pair = {'match': similarity, 'img': img_id, 'letter': letter_id}
		results.append(pair)

	out = sorted(results, key = lambda i: i['match'], reverse=True)

	for each in out:
		print(each)

def test_set():
	results = []
	img_path = './test_img/'
	letter_path = './alphabet/'

	for img_name in os.listdir(img_path):
		if not img_name.startswith('.'):
			img_results = []
			img_id = img_name.replace(".jpg",'')
			# print(img_name)
			img = cv2.imread(img_path + img_name)
			# print(img)
			img_contour = get_bounding_rect_for_isolated_obj_contour(img, img_id)
			img_resized = img_resize(img_contour)

			for cap_letter in os.listdir(letter_path):
				if not cap_letter.startswith('.'):
					letter_id = cap_letter.replace(".png",'')
					letter = cv2.imread(letter_path + cap_letter, cv2.IMREAD_GRAYSCALE)
					similarity = contour_match(img_resized, img_id, letter, letter_id)
					pair = {'match': similarity, 'img': img_id, 'letter': letter_id}
					img_results.append(pair)
			
			sort = sorted(img_results, key = lambda i: i['match'], reverse=True)
			for i in range(5):
				results.append(sort[i])

	count = 0
	for each in results:
		if(count%5==0):
			print("*****************")
		print(each)
		count = count + 1

def test_image_to_image():
	img_path = './test_img/'
	img_list = ['bell.jpg','bowtie_macaroni.jpg','candy_cane.jpg','cheese_v.jpg','christmas_candle.jpg',
				'christmas_sock.jpg','christmas_tree.jpg','curved_macaroni.jpg','ginger_bread_man.jpg',
				'mac_plate_o.jpg','macaroni.jpg','ornament.jpg','pine_cone.jpg','pizza_A.jpg','pizza_V.jpg',
				'present.jpg','rotini.jpg','santa_hat.jpg','wreath.jpg']
	results = []

	for i in range(len(img_list)):
		img1_id = img_list[i].replace('.jpg','')
		img1 = cv2.imread(img_path + img_list[i])
		img1_contour = get_bounding_rect_for_isolated_obj_contour(img1, img1_id)
		img1_resized = img_resize(img1_contour)

		for j in range(i+1,len(img_list)):
			img2_id = img_list[j].replace('.jpg','')
			img2 = cv2.imread(img_path + img_list[j])
			img2_contour = get_bounding_rect_for_isolated_obj_contour(img2, img2_id)
			img2_resized = img_resize(img2_contour)

			similarity = contour_match(img1_resized, img1_id, img2_resized, img2_id)
			pair = {'match': similarity, 'img1': img1_id, 'img2':img2_id}
			results.append(pair)


	sorted_results = sorted(results, key = lambda i: i['match'], reverse=True)

	for each in sorted_results:
		print(each)

#test_one()
#test_set()
test_image_to_image()
