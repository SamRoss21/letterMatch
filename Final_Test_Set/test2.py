import cv2
import numpy as np

output_folder = '/home/ecenaz/research/symbol_finder/letter_match_out/'

def imread_objects_from_path(original_image_path, mask_path):
    img = cv2.imread(original_image_path)
    img_mask = cv2.imread(mask_path, 0) # the 0 should mean to read it in as black/white
    return img, img_mask


# this takes in the original image (imread object) and an alpha mask (probably from the Salient Object Dectecor)
# and finds the bounding rectangle for the object,
# and then writes to file a vew cropped image
# but it should probably just return the image? or take in another arg that seys where to write it to.
def get_bounding_rect_for_isolated_obj(original_image, gray, img_id):
    print("GET BOUNDING RECT ***********************")
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

    cv2.imwrite(output_folder + extracted_file_name, roi)


    cv2.rectangle(original_image,(x,y),(x+w,y+h),(200,0,0),2)
    #cv2.imwrite(output_folder + img_id + '_cont.jpg', original_image)
    return output_folder + extracted_file_name

def get_mask_from_grabcut_result(grabcut_img):
    print('GRABCUT IMG PATH: '+ grabcut_img)
    img = cv2.imread(grabcut_img, cv2.IMREAD_UNCHANGED)
    r_channel, g_channel, b_channel, alpha = cv2.split(img)
    contour_mask = np.zeros(alpha.shape, np.uint8)
    # alpha_not = cv2.bitwise_not(alpha)
    masked_img = cv2.bitwise_or(contour_mask, alpha)
    cv2.imwrite('/home/ecenaz/research/symbol_finder/static/images/grabcut_mask.jpg', masked_img)
    return 'grabcut_mask.jpg'

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
			cv2.imwrite(output_folder +resized_file_name, temp_array2)
		else:
			pixel_line = np.zeros((1, 300), np.uint8)
			temp_array3 = np.concatenate((temp_array2, pixel_line), axis=0)
			cv2.imwrite(output_folder + resized_file_name, temp_array3)


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
			cv2.imwrite(output_folder + resized_file_name, temp_array2)
		else:
			pixel_line = np.zeros((300, 1), np.uint8)
			temp_array3 = np.concatenate((temp_array2, pixel_line), axis=1)
			cv2.imwrite(output_folder +resized_file_name,temp_array3)

	return output_folder + resized_file_name

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
	# print(img1_name)
	img1 = cv2.imread(img1_name,0)
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
	# cnt2 = contours[0]

	#crop tree image
	cropped1 = get_bounding_rect_for_isolated_obj(img1, img1,  img1_id)
	# print(cropped1)
	img3 = cv2.imread(cropped1,0)
	#cv2.imshow('cropped',img3)
	#cv2.waitKey(0)
	#cv2.destroyAllWindows()
	#print ret

	resized1 = img_resize(img1_name, img1_id)
	resized_img1 = cv2.imread(resized1,0)

	#crop triangle image
	cropped2 = get_bounding_rect_for_isolated_obj(img2, img2, img2_id)
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
	return proportion  # img1_id + "+" + img2_id + "," + str(proportion) + ",\r,"

#img name is what the path would be , img_id is wihtout extension etc.
def get_letter_suggestions(img_name, img_id):
    result_file = open(img_id+".txt", "w")
    result_file.write('test_file,letter,score')
    results = []
    print("IN LETTER IMG NAME: " + img_name)
    path = './alphabet/'
    # alphabet_set = [("A.png","A"),("a_lower.png","a"),("B.png","B"),("b_lower.png","b"),
    #             ("C.png","C"),("c_lower.png","c"),("D.png","D"),("d_lower.png","d"),
    #             ("E.png","E"),("e_lower.png","e"),("F.png","F"),("f_lower.png","f"),
    #             ("G.png","G"),("g_lower.png","g"),("H.png","H"),("h_lower.png","h"),
    #             ("I.png","I"),("i_lower.png","i"),("J.png","J"),("j_lower.png","j"),
    #             ("K.png","K"),("k_lower.png","k"),("L.png","L"),("l_lower.png","l"),
    #             ("M.png","M"),("m_lower.png","m"),("N.png","N"),("n_lower.png","n"),
    #             ("O.png","O"),("o_lower.png","o"),("P.png","P"),("p_lower.png","p"),
    #             ("Q.png","Q"),("q_lower.png","q"),("R.png","R"),("r_lower.png","r"),
    #             ("S.png","S"),("s_lower.png","s"),("T.png","T"),("t_lower.png","t"),
    #             ("U.png","U"),("u_lower.png","u"),("V.png","V"),("v_lower.png","v"),
    #             ("W.png","W"),("w_lower.png","w"),("X.png","X"),("x_lower.png","x"),
    #             ("Y.png","Y"),("y_lower.png","y"),("Z.png","Z"),("z_lower.png","z")]
    alphabet_set = [{'files': ['Wide_Filled_a_lower.png', 'Wide_a_lower.png', 'Thin_a_lower.png', 'Thin_Filled_a_lower.png'], 'letter': 'a'}, {'files': ['Wide_Filled_b_lower.png', 'Wide_b_lower.png', 'Thin_b_lower.png', 'Thin_Filled_b_lower.png'], 'letter': 'b'}, {'files': ['Wide_Filled_c_lower.png', 'Wide_c_lower.png', 'Thin_c_lower.png', 'Thin_Filled_c_lower.png'], 'letter': 'c'}, {'files': ['Wide_Filled_d_lower.png', 'Wide_d_lower.png', 'Thin_d_lower.png', 'Thin_Filled_d_lower.png'], 'letter': 'd'}, {'files': ['Wide_Filled_e_lower.png', 'Wide_e_lower.png', 'Thin_e_lower.png', 'Thin_Filled_e_lower.png'], 'letter': 'e'}, {'files': ['Wide_Filled_f_lower.png', 'Wide_f_lower.png', 'Thin_f_lower.png', 'Thin_Filled_f_lower.png'], 'letter': 'f'}, {'files': ['Wide_Filled_g_lower.png', 'Wide_g_lower.png', 'Thin_g_lower.png', 'Thin_Filled_g_lower.png'], 'letter': 'g'}, {'files': ['Wide_Filled_h_lower.png', 'Wide_h_lower.png', 'Thin_h_lower.png', 'Thin_Filled_h_lower.png'], 'letter': 'h'}, {'files': ['Wide_Filled_i_lower.png', 'Wide_i_lower.png', 'Thin_i_lower.png', 'Thin_Filled_i_lower.png'], 'letter': 'i'}, {'files': ['Wide_Filled_j_lower.png', 'Wide_j_lower.png', 'Thin_j_lower.png', 'Thin_Filled_j_lower.png'], 'letter': 'j'}, {'files': ['Wide_Filled_k_lower.png', 'Wide_k_lower.png', 'Thin_k_lower.png', 'Thin_Filled_k_lower.png'], 'letter': 'k'}, {'files': ['Wide_Filled_l_lower.png', 'Wide_l_lower.png', 'Thin_l_lower.png', 'Thin_Filled_l_lower.png'], 'letter': 'l'}, {'files': ['Wide_Filled_m_lower.png', 'Wide_m_lower.png', 'Thin_m_lower.png', 'Thin_Filled_m_lower.png'], 'letter': 'm'}, {'files': ['Wide_Filled_n_lower.png', 'Wide_n_lower.png', 'Thin_n_lower.png', 'Thin_Filled_n_lower.png'], 'letter': 'n'}, {'files': ['Wide_Filled_o_lower.png', 'Wide_o_lower.png', 'Thin_o_lower.png', 'Thin_Filled_o_lower.png'], 'letter': 'o'}, {'files': ['Wide_Filled_p_lower.png', 'Wide_p_lower.png', 'Thin_p_lower.png', 'Thin_Filled_p_lower.png'], 'letter': 'p'}, {'files': ['Wide_Filled_q_lower.png', 'Wide_q_lower.png', 'Thin_q_lower.png', 'Thin_Filled_q_lower.png'], 'letter': 'q'}, {'files': ['Wide_Filled_r_lower.png', 'Wide_r_lower.png', 'Thin_r_lower.png', 'Thin_Filled_r_lower.png'], 'letter': 'r'}, {'files': ['Wide_Filled_s_lower.png', 'Wide_s_lower.png', 'Thin_s_lower.png', 'Thin_Filled_s_lower.png'], 'letter': 's'}, {'files': ['Wide_Filled_t_lower.png', 'Wide_t_lower.png', 'Thin_t_lower.png', 'Thin_Filled_t_lower.png'], 'letter': 't'}, {'files': ['Wide_Filled_u_lower.png', 'Wide_u_lower.png', 'Thin_u_lower.png', 'Thin_Filled_u_lower.png'], 'letter': 'u'}, {'files': ['Wide_Filled_v_lower.png', 'Wide_v_lower.png', 'Thin_v_lower.png', 'Thin_Filled_v_lower.png'], 'letter': 'v'}, {'files': ['Wide_Filled_w_lower.png', 'Wide_w_lower.png', 'Thin_w_lower.png', 'Thin_Filled_w_lower.png'], 'letter': 'w'}, {'files': ['Wide_Filled_x_lower.png', 'Wide_x_lower.png', 'Thin_x_lower.png', 'Thin_Filled_x_lower.png'], 'letter': 'x'}, {'files': ['Wide_Filled_y_lower.png', 'Wide_y_lower.png', 'Thin_y_lower.png', 'Thin_Filled_y_lower.png'], 'letter': 'y'}, {'files': ['Wide_Filled_z_lower.png', 'Wide_z_lower.png', 'Thin_z_lower.png', 'Thin_Filled_z_lower.png'], 'letter': 'z'}, {'files': ['Wide_Filled_A.png', 'Wide_A.png', 'Thin_A.png', 'Thin_Filled_A.png'], 'letter': 'A'}, {'files': ['Wide_Filled_B.png', 'Wide_B.png', 'Thin_B.png', 'Thin_Filled_B.png'], 'letter': 'B'}, {'files': ['Wide_Filled_C.png', 'Wide_C.png', 'Thin_C.png', 'Thin_Filled_C.png'], 'letter': 'C'}, {'files': ['Wide_Filled_D.png', 'Wide_D.png', 'Thin_D.png', 'Thin_Filled_D.png'], 'letter': 'D'}, {'files': ['Wide_Filled_E.png', 'Wide_E.png', 'Thin_E.png', 'Thin_Filled_E.png'], 'letter': 'E'}, {'files': ['Wide_Filled_F.png', 'Wide_F.png', 'Thin_F.png', 'Thin_Filled_F.png'], 'letter': 'F'}, {'files': ['Wide_Filled_G.png', 'Wide_G.png', 'Thin_G.png', 'Thin_Filled_G.png'], 'letter': 'G'}, {'files': ['Wide_Filled_H.png', 'Wide_H.png', 'Thin_H.png', 'Thin_Filled_H.png'], 'letter': 'H'}, {'files': ['Wide_Filled_I.png', 'Wide_I.png', 'Thin_I.png', 'Thin_Filled_I.png'], 'letter': 'I'}, {'files': ['Wide_Filled_J.png', 'Wide_J.png', 'Thin_J.png', 'Thin_Filled_J.png'], 'letter': 'J'}, {'files': ['Wide_Filled_K.png', 'Wide_K.png', 'Thin_K.png', 'Thin_Filled_K.png'], 'letter': 'K'}, {'files': ['Wide_Filled_L.png', 'Wide_L.png', 'Thin_L.png', 'Thin_Filled_L.png'], 'letter': 'L'}, {'files': ['Wide_Filled_M.png', 'Wide_M.png', 'Thin_M.png', 'Thin_Filled_M.png'], 'letter': 'M'}, {'files': ['Wide_Filled_N.png', 'Wide_N.png', 'Thin_N.png', 'Thin_Filled_N.png'], 'letter': 'N'}, {'files': ['Wide_Filled_O.png', 'Wide_O.png', 'Thin_O.png', 'Thin_Filled_O.png'], 'letter': 'O'}, {'files': ['Wide_Filled_P.png', 'Wide_P.png', 'Thin_P.png', 'Thin_Filled_P.png'], 'letter': 'P'}, {'files': ['Wide_Filled_Q.png', 'Wide_Q.png', 'Thin_Q.png', 'Thin_Filled_Q.png'], 'letter': 'Q'}, {'files': ['Wide_Filled_R.png', 'Wide_R.png', 'Thin_R.png', 'Thin_Filled_R.png'], 'letter': 'R'}, {'files': ['Wide_Filled_S.png', 'Wide_S.png', 'Thin_S.png', 'Thin_Filled_S.png'], 'letter': 'S'}, {'files': ['Wide_Filled_T.png', 'Wide_T.png', 'Thin_T.png', 'Thin_Filled_T.png'], 'letter': 'T'}, {'files': ['Wide_Filled_U.png', 'Wide_U.png', 'Thin_U.png', 'Thin_Filled_U.png'], 'letter': 'U'}, {'files': ['Wide_Filled_V.png', 'Wide_V.png', 'Thin_V.png', 'Thin_Filled_V.png'], 'letter': 'V'}, {'files': ['Wide_Filled_W.png', 'Wide_W.png', 'Thin_W.png', 'Thin_Filled_W.png'], 'letter': 'W'}, {'files': ['Wide_Filled_X.png', 'Wide_X.png', 'Thin_X.png', 'Thin_Filled_X.png'], 'letter': 'X'}, {'files': ['Wide_Filled_Y.png', 'Wide_Y.png', 'Thin_Y.png', 'Thin_Filled_Y.png'], 'letter': 'Y'}, {'files': ['Wide_Filled_Z.png', 'Wide_Z.png', 'Thin_Z.png', 'Thin_Filled_Z.png'], 'letter': 'Z'}]
    for obj in alphabet_set:
        letter = obj["letter"]
        for file in object["files"]:
            match = test_pair(img_name,img_id, path + file, letter)
        # print(match)

        result_file.write()
        # results.write("test_file:,"+test+", letter:," + letter_name +", score:," + str(match*100)+",\n")


        
    return sorted(results, key =  lambda i: i['score'], reverse=True)
    # return results
	

##########################################################
# results = open("results.txt","a")
# test_set = [("antlers.png","antlers"),("bauble.png","bauble"),("bell.png","bell"),("boot.png","boot"),
# 			("clover.png","clover"),("deer.png","deer"), ("flowers.png","flowers"),("gourd.png","gourd"),
# 			("lamb.png","lamb"),("palm.png","palm"), ("pine_cone.png","pine_cone"),("pine.png","pine"),
# 			("pizza.png","pizza"),("present.png","present"), ("shoe.png","shoe"),("shovel.png","shovel"),
# 			("sundae.png","sundae"),("sweater.png","sweater"),("tree.jpeg","tree"),("wreath.png","wreath")]

# alphabet_set = [("A.png","A"),("a_lower.png","a"),("B.png","B"),("D.png","D"),("I.png","I"),
# 				("i_lower.png","i"),("L.png","L"),("n.png","n"),("O.png","O"),
# 				("o_lower.png","o"),("p_lower.png","p"),("P.png","P"),("T.png","T")]

# for image in test_set:
# 	for letter in alphabet_set:
# 		print(image[0],image[1],letter[0],letter[1])
# 		results.write(test_pair(image[0],image[1],letter[0],letter[1]))





import sys
import os
from string import ascii_lowercase
from string import ascii_uppercase
if __name__ == '__main__':
	# results = open("results.txt","w")
    # path = '/home/ecenaz/research/symbol_finder/static/full_alphabet/'
    # # img_name = sys.argv[1]
    # # img_id = sys.argv[2]
    # results_list = []
    # for letter_tuple in alphabet_set:
    #     match = test_pair(img_name,img_id,path +letter_tuple[0], letter_tuple[1])
    #     # results.write(test_pair(image[0],image[1],letter[0],letter[1]))
    #     # if match >= 0.5:
    #     results.write("letter:," + letter_tuple[1] +", score:," + str(match*100)+",\n")
    #     results_list.append({'letter':letter_tuple[1], 'score': (match*100)})
    #     print( sorted(results_list, key =  lambda i: i['score'], reverse=True))
    # print get_letter_suggestions("./test_set/santa_hat.jpg", "santa_hat")
    # groups = ['./Thin', "./Thin_Filled", "./Wide","./Wide_Filled"]

	# for test in os.listdir('./test_set'):
	# 	results = open(test+".txt", "w")
	# 	for group in groups:
	# 		for filename in os.listdir(group):
	# 			if not filename.startswith('.'):
	# 				img_name = "./test_set/" + test
	# 				img_id = test.replace('.jpg', '')
	# 				letter_img_path = group + "/"+ filename
	# 				letter_name = filename.replace(".png", "")
	# 				match = test_pair(letter_img_path, letter_name, img_name,img_id)
	# 				results.write("test_file:,"+test+", letter:," + letter_name +", score:," + str(match*100)+",\n")
    alphabet = []
    file_names = ["Wide_Filled_", "Wide_", "Thin_", "Thin_Filled_"]
    for c in ascii_lowercase:
        letter_obj = {
            "letter" : c,
            "files" : []
        }
        for name in file_names:
            letter_obj['files'].append(name + c + "_lower.png")
        alphabet.append(letter_obj)
    for c in ascii_uppercase: 
        letter_obj = {
            "letter" : c,
            "files" : []
        }
        for name in file_names:
            letter_obj['files'].append(name + c + ".png")
        alphabet.append(letter_obj)
    print(alphabet)
    results = open("alphabet.txt","w")
    results.write(str(alphabet))

    




