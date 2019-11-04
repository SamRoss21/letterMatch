import cv2
import numpy as np
import os

output_folder = '/Users/samross/Desktop/Main/Barnard/Research/Shape_Matching/OpenCV_Testing/Main_Test_Set/contour_match/out/'

def get_bounding_rect_for_isolated_obj_contour(gray, output_folder, img_id):
  imgray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
  ret, thresh = cv2.threshold(imgray, 127, 255, 0)
  contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

  gray_size = gray.shape[0] * gray.shape[1]

  # Find object with the biggest bounding box
  mx = (0,0,0,0)      # biggest bounding box so far
  mx_area = 0
  winning_contour = None
  for cont in contours:
     x,y,w,h = cv2.boundingRect(cont)
     area = w*h
     if area == gray_size:
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

  return contour_mask

def test_contour():
  path = '/Users/samross/Desktop/Main/Barnard/Research/Shape_Matching/OpenCV_Testing/Main_Test_Set/contour_match/test_img/'
  img_path = path + 'bowtie_macaroni.jpg'
  img_id = 'bowtie'

  img = cv2.imread(img_path)

  name = get_bounding_rect_for_isolated_obj_contour(img, img, output_folder, img_id)

test_contour()

