import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import argparse
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
from moviepy.editor import VideoFileClip
from IPython.display import HTML
def get_arguments():
    ap = argparse.ArgumentParser()

    ap.add_argument("-d", "--diagnostics", required=False, default='False', type=str,
                    help="Show additional diagnostics on video.")
    ap.add_argument("-v", "--visualization", required=False, default='False', type=str,
                    help="Visualize data only and do not create videos.")
    ap.add_argument("-c", "--challenge", required=False, default='False', type=str,
                    help="Process just the challenge video.")
    ap.add_argument("-s", "--skipharderchallenge", required=False, default='False', type=str,
                    help="Do the basic and challenge videos but skip the harder challenge.")
    args = vars(ap.parse_args())
    return args

def get_boolean_arg(args,arg_name):
    if args[arg_name] == "True" or args[arg_name] == "true":
        boolean_arg=True
    elif args[arg_name] == "False" or args[arg_name] == "false":
        boolean_arg=False
    return boolean_arg

class Line():
    def __init__(self):
        self.num_to_avg=5
        self.num_pts=720
        self.detected = False
        self.bestx = np.array([self.num_to_avg,self.num_pts],dtype=np.int32)
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        self.best_real_world_fit = None
        #polynomial coefficients for the most recent fit
        self.current_fit=[]
        self.current_real_world_fit=[]
        #radius of curvature of the line in some units
        self.radius_of_curvature = None
        self.diffs = np.array([0,0,0], dtype='float')

    def track_fit(self, h, fit, real_world_fit):
        # add a found fit to the line, up to n
        if fit is not None:
            if self.best_fit is not None and len(self.current_fit) > 0:
                fit_x_int = int(fit[0] * h ** 2 + fit[1] * h + fit[2])
                best_fit_int = int(self.best_fit[0] * h ** 2 + self.best_fit[1] * h + self.best_fit[2])
                x_int_diff = abs(fit_x_int - best_fit_int)
            else:
                x_int_diff=0

            if x_int_diff > 100:
                self.detected = False
            else:
                self.detected = True
                self.current_fit.append(fit)
                self.current_real_world_fit.append((real_world_fit))
                if len(self.current_fit) > self.num_to_avg:
                    self.current_fit = self.current_fit[len(self.current_fit) - self.num_to_avg:]
                    self.current_real_world_fit = self.current_real_world_fit[len(self.current_real_world_fit) - self.num_to_avg:]
                self.best_fit = np.average(self.current_fit, axis=0)
                self.best_real_world_fit = np.average(self.current_real_world_fit, axis=0)
        # remove a fit from the history if we are unable to find a fit this time
        else:
            self.detected = False
            if len(self.current_fit) > 0:
                # throw out the oldest fit
                self.current_fit = self.current_fit[:len(self.current_fit) - 1]
                self.current_real_world_fit=self.current_real_world_fit[:len(self.current_real_world_fit) - 1]
            if len(self.current_fit) > 0:
                # Calculate the average of the remaining fits
                self.best_fit = np.average(self.current_fit, axis=0)
                self.best_real_world_fit = np.average(self.current_real_world_fit, axis=0)
    def invalidate_last_fit(self):
        self.detected = False
        if self.current_fit is not None:
            if len(self.current_fit) > 0:
                del self.current_fit[-1]
                del self.current_real_world_fit[-1]
                if len(self.current_fit) > 0:
                    self.best_fit = np.average(self.current_fit, axis=0)
                    self.best_real_world_fit = np.average(self.current_real_world_fit, axis=0)

def process_2files_in_directory(directory_name,proc,description):
    images = glob.glob(directory_name)
    imgs = []
    processed_imgs = []
    titles1 = []
    titles2 = []
    for fname in images:
        img = mpimg.imread(fname)
# if the image has an alpha channel, get rid of it
        if img.shape[2] == 4:
            img=img[:,:,:3]
        processed_img=proc(img)
        imgs.append(img)
        titles1.append((fname))
        processed_imgs.append((processed_img))
        titles2.append(fname + description)
    return imgs, titles1, processed_imgs, titles2

def process_3files_in_directory(directory_name, proc, descr1, descr2):
    images = glob.glob(directory_name)
    imgs=[]
    processed_imgs1=[]
    processed_imgs2=[]
    titles1=[]
    titles2=[]
    titles3=[]
    for fname in images:
        img = mpimg.imread(fname)
# if the image has an alpha channel, get rid of it
        if img.shape[2] == 4:
            img=img[:,:,:3]
        proc_img1, proc_img2=proc(img)
        imgs.append(img)
        titles1.append((fname))
        processed_imgs1.append((proc_img1))
        titles2.append(fname + descr1)
        processed_imgs2.append((proc_img2))
        titles3.append(fname + descr2)
    return imgs, titles1, processed_imgs1, titles2, processed_imgs2, titles3

# This function is used for data visualization
def maskedWarpedImage(img):
    kernel_size = 5
    blurred_img = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
    masked_img, warped_img = get_masked_warped_image(blurred_img)
    return masked_img, warped_img

# This function is used for data visualization
def warpPerspective(img):
    imshape = img.shape
    warped_img = cv2.warpPerspective(img, M, (imshape[1], imshape[0]))
    return warped_img

# This function is used for data visualization
def get_perspective_transform_images(directory_name):
    proc=warpPerspective
    imgs, titles1, warped_imgs, titles2=process_2files_in_directory(directory_name,proc,"warped")
    return imgs,titles1,warped_imgs,titles2

# This function is used for data visualization
def get_undistorted_images(directory_name):
    proc=undistort
    imgs, titles1, undistorted_imgs, titles2 = process_2files_in_directory(directory_name, proc, "undistorted")
    return imgs,titles1,undistorted_imgs,titles2

# This function is used for data visualization
def get_masked_warped_images(directory_name):
    proc=maskedWarpedImage
    imgs, titles1, masked_imgs, titles2, warped_imgs, titles3=process_3files_in_directory(directory_name, proc,"masked","warped")
    return imgs, titles1, masked_imgs, titles2, warped_imgs, titles3

# function is used to draw figures for data visualization
def show_images_side_by_side(img1,title1,img2,title2,is_gray=False):
    fig = plt.figure(figsize=(6, 4))
    subfig1 = fig.add_subplot(1, 2, 1)
    subfig1.imshow(img1)
    subfig1.set_title(title1, fontsize=20)
    subfig2 = fig.add_subplot(1, 2, 2)
    if is_gray:
        subfig2.imshow(img2)
    else:
        subfig2.imshow(img2, cmap='gray')
    subfig2.set_title(title2, fontsize=20)
    fig.tight_layout()
    plt.show()
    return subfig1, subfig2

# function is used to draw figures for data visualization
def show_array_2images_side_by_side(imgs1,titles1,imgs2,titles2,is_gray=False):
    assert(len(imgs1)==len(imgs2))
    assert(len(imgs1)==len(titles1))
    assert(len(titles1)==len(titles2))
    n=len(imgs1)
    h_figure=2*n
    fig = plt.figure(figsize=(6, h_figure))
    i=0
    num_cols=2
    for img in imgs1:
        subfig1 = fig.add_subplot(n, num_cols, num_cols*i+1)
        subfig1.imshow(img)
        subfig1.set_title(titles1[i], fontsize=5)

        subfig2 = fig.add_subplot(n, num_cols, num_cols*i+2)
        if is_gray:
            subfig2.imshow(imgs2[i],cmap='gray')
        else:
            subfig2.imshow(imgs2[i])
        subfig2.set_title(titles2[i], fontsize=5)
        i+=1
    fig.tight_layout()
    plt.show()
    return fig

# function is used to draw figures for data visualization
def show_array_3images_side_by_side(imgs1,titles1,imgs2,titles2,imgs3,titles3,is_gray=False):
    assert(len(imgs1)==len(imgs2))
    assert (len(imgs2) == len(imgs3))
    assert(len(imgs3)==len(titles1))
    assert(len(titles2)==len(titles1))
    assert (len(titles3) == len(titles2))
    n=len(imgs1)
    h_figure = 2 * n
    fig = plt.figure(figsize=(9, h_figure))
    i=0
    num_cols=3
    for img in imgs:
        subfig1 = fig.add_subplot(n, num_cols, num_cols*i+1)
        subfig1.imshow(img)
        subfig1.set_title(titles1[i], fontsize=5)
        subfig2 = fig.add_subplot(n, num_cols, num_cols*i+2)
        if is_gray:
            subfig2.imshow(imgs2[i],cmap='gray')
        else:
            subfig2.imshow(imgs2[i])
        subfig2.set_title(titles2[i], fontsize=5)
        subfig3 = fig.add_subplot(n, num_cols, num_cols * i + 3)
        if is_gray:
            subfig3.imshow(imgs3[i], cmap='gray')
        else:
            subfig3.imshow(imgs3[i])
        subfig3.set_title(titles3[i], fontsize=5)
        i+=1
    fig.tight_layout()
    plt.show()
    return fig

# function is used to draw figures for data visualization
def view_histogram(data,plot_name,file_name):
    fig = plt.figure(figsize=(8, 5))
    plt.plot(data)
    plt.xlim(0, 1280)
    plt.title(plot_name)
    plt.ylabel('# of Pixels')
    plt.show()
    fig.savefig(file_name,bbox_inches='tight')

# function is used to draw figures for data visualization
def draw_windows(img,binary_warped,title,filename):
    left_fit, right_fit, left_real_world, right_real_world, rectangles = get_left_and_right_lane_fits(img,
        binary_warped)
    left_fitx=None
    right_fitx=None

    left_lane_inds = []
    right_lane_inds = []

    out_img=np.array(img.shape)
    out_img = np.uint8(np.dstack((binary_warped, binary_warped, binary_warped))*255)

    nonzero = out_img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    # Iterate through the rectangles returned by get_left_and_right_lane_fits
    for rect in rectangles:
        # Identify window boundaries in x and y (and right and left)
        cv2.rectangle(out_img, (rect[2], rect[0]), (rect[3], rect[1]), (0, 255, 0), 2)
        cv2.rectangle(out_img, (rect[4], rect[0]), (rect[5], rect[1]), (0, 255, 0), 2)
        good_left_inds = ((nonzeroy >= rect[0]) & (nonzeroy < rect[1]) & (nonzerox >= rect[2]) & (
        nonzerox < rect[3])).nonzero()[0]
        good_right_inds = ((nonzeroy >= rect[0]) & (nonzeroy < rect[1]) & (nonzerox >= rect[4]) & (
        nonzerox < rect[5])).nonzero()[0]
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)
    ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
    if left_fit is not None:
        left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    if right_fit is not None:
        right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    fig = plt.figure(figsize=(8, 5))
    plt.imshow(out_img)
    if left_fit is not None:
        plt.plot(left_fitx, ploty, color='yellow')
    if right_fit is not None:
        plt.plot(right_fitx, ploty, color='yellow')

    plt.xlim(0, 1280)
    plt.ylim(720, 0)
    plt.title(title)
    plt.show()
    fig.savefig(filename,bbox_inches='tight')

# function is used to draw figures for data visualization
def visualize_search_windows(img,warped_img,title,filename):
    left_fit, right_fit, left_real_world, right_real_world, rectangles = get_left_and_right_lane_fits(img,warped_img)
    out_img=draw_windows(img,warped_img,title,filename)

# get the channels used for our color thresholds
def convert_image2(img):
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    luv = cv2.cvtColor(img, cv2.COLOR_RGB2Luv)
    lab=cv2.cvtColor(img, cv2.COLOR_RGB2Lab)
    s_channel = hls[:, :, 2]
    l_channel = luv[:, :, 0]
    b_channel = lab[:, :, 2]
    return s_channel, l_channel, b_channel

def convert_image(img):
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    l_channel = hls[:, :, 1]
    s_channel = hls[:, :, 2]
    return s_channel

def convert_image3(img):
    hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    v_channel = hsv_img[:, :, 2]
    return v_channel

def calibrate_using_chessboard_images(sizex,sizey,directory_wildcard_filename):
    images = glob.glob(directory_wildcard_filename)
    num_good=0
    nx=sizex
    ny=sizey
    objpoints=[]
    imgpoints=[]
    orig_imgs=[]
    titles1=[]
    titles2=[]
    processed_imgs=[]
    objp=np.zeros((nx*ny,3),np.float32)
    objp[:,:2]=np.mgrid[0:nx,0:ny].T.reshape(-1,2)

    for fname in images:
        img=mpimg.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret,corners = cv2.findChessboardCorners(gray, (nx,ny), None)
        if ret==True:
            orig_imgs.append(img.copy())
            num_good+=1
            imgpoints.append(corners)
            objpoints.append(objp)
            processed_img=cv2.drawChessboardCorners(img,(nx,ny),corners,ret)

            titles1.append(fname)
            processed_imgs.append(processed_img)
            titles2.append(fname+"processed")
    print("Number of good images={}".format(num_good))
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    return ret, mtx, dist, rvecs, tvecs, orig_imgs, titles1, processed_imgs, titles2

def mag_thresh(img, sobel_kernel=5, mag_thresh=(25, 255)):
    # Convert to grayscale
    single_channel_img = convert_image3(img)
    # Take both Sobel x and y gradients
    sobelx = cv2.Sobel(single_channel_img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(single_channel_img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Calculate the gradient magnitude
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    # Rescale to 8 bit
    scale_factor = np.max(gradmag)/255
    gradmag = (gradmag/scale_factor).astype(np.uint8)
    # Create a binary image of ones where threshold is met, zeros otherwise
    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1
    # Return the binary image
    return binary_output

def dir_threshold(img, sobel_kernel=5, thresh=(0, np.pi/4)):
    # Grayscale
    single_channel_img = convert_image3(img)
    # Calculate the x and y gradients
    sobelx = cv2.Sobel(single_channel_img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(single_channel_img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Take the absolute value of the gradient direction,
    # apply a threshold, and create a binary image result
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    binary_output =  np.zeros_like(absgraddir)
    binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1
    # Return the binary image
    return binary_output

def abs_sobel_thresh(img, orient='x', sobel_kernel=5, thresh=(25, 255)):
    # Convert to grayscale
    single_channel_img = convert_image3(img)
    # Apply x or y gradient with the OpenCV Sobel() function
    # and take the absolute value
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(single_channel_img, cv2.CV_64F, 1, 0,ksize=sobel_kernel))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(single_channel_img, cv2.CV_64F, 0, 1,ksize=sobel_kernel))
    # Rescale back to 8 bit integer
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # Create a copy and apply the threshold
    binary_output = np.zeros_like(scaled_sobel)
    thresh_min=thresh[0]
    thresh_max=thresh[1]
    # Here I'm using inclusive (>=, <=) thresholds, but exclusive is ok too
    binary_output[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1
    # Return the result
    return binary_output

def color_threshold(img):
    # Threshold color channel
    s_thresh_min = 180
    s_thresh_max = 255
    l_thresh_min = 220
    l_thresh_max = 255
    b_thresh_min = 155
    b_thresh_max = 200

    s_channel, l_channel, b_channel=convert_image2(img)

    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel > s_thresh_min) & (s_channel <= s_thresh_max)] = 1

    l_binary = np.zeros_like(l_channel)
    l_binary[(l_channel > l_thresh_min) & (l_channel <= l_thresh_max)] = 1

    b_binary = np.zeros_like(b_channel)
    b_binary[(b_channel > b_thresh_min) & (b_channel <= b_thresh_max)] = 1

    combined_binary = np.zeros_like(l_channel)
    combined_binary[(l_binary == 1) | (b_binary == 1)] = 1
    return combined_binary

def apply_thresholds(img):
    grad_binary=calculate_gradiant(img)
    color_binary=color_threshold(img)
    combined_binary = np.zeros_like(grad_binary)
    combined_binary[(grad_binary == 1) | (color_binary == 1)] = 1
#    combined_binary[(color_binary == 1)] = 1
#    combined_binary[(grad_binary == 1)] = 1
    return combined_binary

def apply_thresholds2(img_unwarp):
    s_thresh_min = 170
    s_thresh_max = 255
    l_thresh_min = 220
    l_thresh_max = 255
    b_thresh_min = 190
    b_thresh_max = 255

    s_channel, l_channel, b_channel = convert_image2(img_unwarp)
# we ignore the S channel
    l_channel = l_channel * (255 / np.max(l_channel))
    if np.max(b_channel) > 175:
        b_channel = b_channel * (255 / np.max(b_channel))
    l_output = np.zeros_like(l_channel)
    l_output[(l_channel > l_thresh_min) & (l_channel <= l_thresh_max)] = 1

    b_output = np.zeros_like(b_channel)
    b_output[((b_channel > b_thresh_min) & (b_channel <= b_thresh_max))] = 1

    combined = np.zeros_like(b_output)
    combined[(l_output == 1) | (b_output == 1)] = 1
    return combined

def calculate_gradiant(img):
    kernel_size=7

    # Run the function
    gradx = abs_sobel_thresh(img, orient='x', sobel_kernel=kernel_size, thresh=(25, 255))
    grady = abs_sobel_thresh(img, orient='y', sobel_kernel=kernel_size, thresh=(25, 255))
    mag_binary = mag_thresh(img, sobel_kernel=kernel_size, mag_thresh=(25, 255))
    dir_binary = dir_threshold(img, sobel_kernel=kernel_size, thresh=(.65, 1.05))
    # Combine all the thresholding information
    combined = np.zeros_like(dir_binary)
#    combined[((gradx == 1) | (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1
    combined[((gradx == 1) ) | ((mag_binary == 1) )] = 1
#    combined[((gradx == 1) | (grady == 1))] = 1
#    combined[(gradx == 1)] = 1
    return combined

def undistort (img):
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    return undist

def region_of_interest(img, vertices):
    """"
    Applies an image mask.
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    # defining a blank mask to start with
    mask = np.zeros_like(img)

    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def region_of_interest_with_transform(img, vertices):
    """"
    Applies an image mask.
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    # defining a blank mask to start with
    mask = np.zeros_like(img)

    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    mask=warpPerspective(mask)
    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def setup_warp_transform(img):
    src = np.array([[(529, 466), (751, 466),
                       (1218, 675), (62, 675)]], dtype=np.float32)
    dst = np.array([[(139, 0), (1141, 0),
                       (1141, 720), (139, 720)]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(src, dst)
    return M

def setup_inverse_warp_transform(img):
    src = np.array([[(529, 466), (751, 466),
                      (1218, 675), (62, 675)]], dtype=np.float32)
    dst = np.array([[(139, 0), (1141, 0),
                     (1141, 720), (139, 720)]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(dst, src)
    return M

def get_masked_warped_image(img):
    imshape = img.shape
    vertices = np.array([[(9 / 20 * imshape[1], 6 / 10 * imshape[0]), (11 / 20 * imshape[1], 6 / 10 * imshape[0]),
                          (9 / 10 * imshape[1], imshape[0]), (1 / 10 * imshape[1], imshape[0])]], dtype=np.int32)
    img=undistort(img)
#    masked_img = region_of_interest(img, vertices)
#    warped_img = cv2.warpPerspective(masked_img, M, (imshape[1], imshape[0]))
    warped_img = cv2.warpPerspective(img, M, (imshape[1], imshape[0]))
    binary_warped_img=apply_thresholds(warped_img)
#    binary_warped_img = region_of_interest_with_transform(binary_warped_img, vertices)
    return warped_img, binary_warped_img

def draw_lane_on_img(original_img, binary_img, left_path_coordinates, right_path_coordinates, Minv):
    new_img = np.copy(original_img)
    h= original_img.shape[0]
    w=original_img.shape[1]
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(binary_img).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Draw the lane onto the warped blank image
    pts = np.vstack((left_path_coordinates, np.flipud(right_path_coordinates)))
    cv2.fillPoly(color_warp, np.int32([pts]), (0, 255, 0))
    cv2.polylines(color_warp, ([left_path_coordinates]), isClosed=False, color=(255, 0, 255), thickness=15)
    cv2.polylines(color_warp, ([right_path_coordinates]), isClosed=False, color=(0, 255, 255), thickness=15)

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    new_warp = cv2.warpPerspective(color_warp, Minv, (w, h))
    # Combine the result with the original image
    result = cv2.addWeighted(new_img, 1, new_warp, 0.3, 0)
    return result

def get_xaxis_histogram(img):
    histogram = np.sum(img, axis=0)
    return histogram

def get_left_and_right_lane_fits(img,warped_img):
    ym_per_pix = 30 / 720  # meters per pixel in y dimension
    xm_per_pix = 3.7 / 700  # meters per pixel in x dimension
    height = warped_img.shape[0]
    width = warped_img.shape[1]
# Choose the number of sliding windows
    nwindows = 9

    bottom_half_img = warped_img[360:, :]
    histogram=get_xaxis_histogram(bottom_half_img)

    nonzero = bottom_half_img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    one_third_point = np.int(histogram.shape[0] // 3)
    two_third_point = np.int(2*histogram.shape[0] // 3)
    leftx_base = np.argmax(histogram[:one_third_point])
    rightx_base = np.argmax(histogram[two_third_point:]) + two_third_point

    # Set height of windows
    window_height = warped_img.shape[0] // nwindows
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = warped_img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50

    left_lane_inds = []
    right_lane_inds = []

    rectangles = []
    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = warped_img.shape[0] - (window + 1) * window_height
        win_y_high = warped_img.shape[0] - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        rectangles.append((win_y_low, win_y_high, win_xleft_low, win_xleft_high, win_xright_low, win_xright_high))
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (
        nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (
        nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit a second order polynomial to each
    if len(lefty) > 0:
        left_poly_coeffs = np.polyfit(lefty, leftx, 2)
        left_poly_coeffs_real_world = np.polyfit(lefty * ym_per_pix, leftx * xm_per_pix, 2)
    else:
        print("Cannot do left lane fit so setting fits to None. For image={}".format(num_times_called))
        left_poly_coeffs=None
        left_poly_coeffs_real_world=None

    if len(righty) > 0:
        right_poly_coeffs = np.polyfit(righty, rightx, 2)
        right_poly_coeffs_real_world = np.polyfit(righty * ym_per_pix, rightx * xm_per_pix, 2)
    else:
        print("Cannot do right lane fit so setting fits to None. For image={}".format(num_times_called))
        right_poly_coeffs = None
        right_poly_coeffs_real_world=None

    right_line.track_fit(height,right_poly_coeffs,right_poly_coeffs_real_world)

    if right_line.best_fit is not None:
        right_poly_coeffs=right_line.best_fit
        right_poly_coeffs_real_world=right_line.best_real_world_fit
    else:
        print("No best fit available for right lane")

    left_line.track_fit(height,left_poly_coeffs,left_poly_coeffs_real_world)
    if left_line.best_fit is not None:
        left_poly_coeffs=left_line.best_fit
        left_poly_coeffs_real_world=left_line.best_real_world_fit
    else:
        print("No best fit available for left lane")

    return left_poly_coeffs, right_poly_coeffs, left_poly_coeffs_real_world, right_poly_coeffs_real_world, rectangles

def get_left_and_right_lane_lines_from_previous_fit(binary_warped,left_fit,right_fit):
    height = binary_warped.shape[0]
    width = binary_warped.shape[1]
    ym_per_pix = 30 / 720  # meters per pixel in y dimension
    xm_per_pix = 3.7 / 700  # meters per pixel in x dimension
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    margin = 100
    left_lane_inds = ((nonzerox > (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy + left_fit[2] - margin)) & (
    nonzerox < (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy + left_fit[2] + margin)))
    right_lane_inds = (
    (nonzerox > (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy + right_fit[2] - margin)) & (
    nonzerox < (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy + right_fit[2] + margin)))

    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    # Fit a second order polynomial to each
    if len(lefty) > 0:
        left_poly_coeffs = np.polyfit(lefty, leftx, 2)
        left_poly_coeffs_real_world = np.polyfit(lefty * ym_per_pix, leftx * xm_per_pix, 2)
    else:
        print("Cannot do left lane fit so setting fits to None. For image={}".format(num_times_called))
        left_poly_coeffs = None
        left_poly_coeffs_real_world = None
    left_line.track_fit(height,left_poly_coeffs, left_poly_coeffs_real_world)
    if left_line.best_fit is not None:
        left_poly_coeffs = left_line.best_fit
        left_poly_coeffs_real_world = left_line.best_real_world_fit
    else:
        print("No best fit available for left lane")

    if len(righty) > 0:
        right_poly_coeffs = np.polyfit(righty, rightx, 2)
        right_poly_coeffs_real_world = np.polyfit(righty * ym_per_pix, rightx * xm_per_pix, 2)
    else:
        print("Cannot do right lane fit so setting fits to None. For image={}".format(num_times_called))
        right_poly_coeffs = None
        right_poly_coeffs_real_world = None
    right_line.track_fit(height,right_poly_coeffs, right_poly_coeffs_real_world)
    if right_line.best_fit is not None:
        right_poly_coeffs = right_line.best_fit
        right_poly_coeffs_real_world = right_line.best_real_world_fit
    else:
        print("No best fit available for right lane")
    return left_poly_coeffs, right_poly_coeffs, left_poly_coeffs_real_world, right_poly_coeffs_real_world

def get_left_and_right_lane_lines(left_poly_coeffs,right_poly_coeffs,img):
    h=img.shape[0]
    w=img.shape[1]
    y = np.linspace(0, h - 1, num=h)
    left_polyx = left_poly_coeffs[0] * y ** 2 + left_poly_coeffs[1] * y + left_poly_coeffs[2]
    right_polyx = right_poly_coeffs[0] * y ** 2 + right_poly_coeffs[1] * y + right_poly_coeffs[2]
    left_path_coordinates = tuple(zip(left_polyx, y))
    right_path_coordinates = tuple(zip(right_polyx, y))
    left_path_coordinates = np.array(left_path_coordinates, dtype=np.int32)
    right_path_coordinates = np.array(right_path_coordinates, dtype=np.int32)
    return left_path_coordinates, right_path_coordinates

def get_curvature(y_eval,left_fit_cr,right_fit_cr):
    ym_per_pix = 30/720 # meters per pixel in y dimension

    # Calculate the new radii of curvature
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    # Now our radius of curvature is in meters
    return left_curverad, right_curverad

def get_distance_from_lane_center(img,left_path_coordinates, right_path_coordinates):
    # Find the position of the car from the center
    # It will show if the car is 'x' meters from the left or right
    position = img.shape[1]/2
    bottom=700
    xleft=left_path_coordinates[left_path_coordinates[:,1]>bottom][:,0]
    xright=right_path_coordinates[right_path_coordinates[:, 1] > bottom][:, 0]

    left=np.min(xleft)
    right=np.max(xright)

    center = (left + right)/2
    # Define conversions in x and y from pixels space to meters
    xm_per_pix = 3.7/700 # meters per pixel in x dimension
    return (position - center)*xm_per_pix

def draw_text_onto_image(img, curve_radius, distance_from_center):
    annotated_img = np.copy(img)
    h = annotated_img.shape[0]
    font = cv2.FONT_HERSHEY_SIMPLEX
    text = 'Curve radius: ' + '{:04.2f}'.format(curve_radius) + 'm'
    cv2.putText(annotated_img, text, (200,70), font, 1.5, (200,255,155), 2, cv2.LINE_AA)
    direction = ''
    if distance_from_center > 0:
        direction = 'right of '
    elif distance_from_center < 0:
        direction = 'left of '
    else:
        direction='on '
    abs_center_dist = abs(distance_from_center)
    text = '{:04.3f}'.format(distance_from_center) + 'm ' + direction + ' center'
    cv2.putText(annotated_img, text, (200,120), font, 1.5, (200,255,155), 2, cv2.LINE_AA)
    return annotated_img

def draw_diagnostic_text_onto_image(img, l_fit_x_int, r_fit_x_int,x_int_diff,num_called,newWindows,curvatures):
    annotated_img = np.copy(img)
    h = annotated_img.shape[0]
    font = cv2.FONT_HERSHEY_SIMPLEX
    text = 'Left Intercept: ' + '{}'.format(l_fit_x_int) + ' pixels'
    cv2.putText(annotated_img, text, (200,170), font, 1.5, (200,255,155), 2, cv2.LINE_AA)

    text = 'Right Intercept: ' + '{}'.format(r_fit_x_int) + ' pixels'
    cv2.putText(annotated_img, text, (200,220), font, 1.5, (200,255,155), 2, cv2.LINE_AA)

    text = 'Diff: ' + '{}'.format(x_int_diff) + ' pixels'
    cv2.putText(annotated_img, text, (200, 270), font, 1.5, (200, 255, 155), 2, cv2.LINE_AA)

    text = 'Index in video: ' + '{}'.format(num_called)
    cv2.putText(annotated_img, text, (200, 320), font, 1.5, (200, 255, 155), 2, cv2.LINE_AA)

    if newWindows==True:
        text="New windows created"
        cv2.putText(annotated_img, text, (200, 370), font, 1.5, (200, 0, 0), 2, cv2.LINE_AA)
    else:
        text="Using previous fit"
        cv2.putText(annotated_img, text, (200, 370), font, 1.5, (200, 255, 155), 2, cv2.LINE_AA)

    curr_y=420
    for left_curvature, right_curvature in curvatures:
        text = 'Left curvature={} Right curvature={}'.format(int(left_curvature),int(right_curvature))
        cv2.putText(annotated_img, text, (200, curr_y), font, 1.5, (200, 255, 155), 2, cv2.LINE_AA)
        curr_y+=50
    return annotated_img

def process_lines(left_poly_coeffs, right_poly_coeffs,left_path_coordinates, right_path_coordinates ,avg_curvature,distance_from_lane_center):
    if left_path_coordinates is not None:
        left_line.detected=True
    if right_path_coordinates is not None:
        right_line.detected=True

def run_sanity_check(img,left_fit_coeffs, right_fit_coeffs,left_real_world, right_real_world):
    check_passed=True
    h=img.shape[0]
    l_fit_x_int = 0
    r_fit_x_int = 0
    x_int_diff = r_fit_x_int - l_fit_x_int
    left_curvature=0
    right_curvature=0
    curvatures=[]
    if left_fit_coeffs is not None and right_fit_coeffs is not None:
        # Check distance between x_intercepts
        h = img.shape[0]
        l_fit_x_int = int(left_fit_coeffs[0] * h ** 2 + left_fit_coeffs[1] * h + left_fit_coeffs[2])
        r_fit_x_int = int(right_fit_coeffs[0] * h ** 2 + right_fit_coeffs[1] * h + right_fit_coeffs[2])
        x_int_diff = abs(r_fit_x_int - l_fit_x_int)
        if x_int_diff < 200 or x_int_diff > 800:
            check_passed=False
        # check curvatures
        test_multiplier=2

        left_curvature, right_curvature = get_curvature(h / 2, left_real_world, right_real_world)
        curvatures.append((left_curvature, right_curvature))
        avg_curvature = (left_curvature + right_curvature) / 2
        if left_curvature > right_curvature:
            ratio=left_curvature/right_curvature
        else:
            ratio=right_curvature/left_curvature
        if avg_curvature < 3000 and abs(ratio) > 8:
            check_passed=False
    else:
        check_passed=False
    return check_passed,l_fit_x_int,r_fit_x_int,x_int_diff, curvatures

def do_pipeline(img,img_index):
    kernel_size = 3
    left_poly_coeffs = None
    right_poly_coeffs = None
    blurred_img = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
    masked_img, warped_img = get_masked_warped_image(blurred_img)
    if not left_line.detected or not right_line.detected:
        left_poly_coeffs, right_poly_coeffs, left_real_world, right_real_world, rectangles = get_left_and_right_lane_fits(img,
                                                                                                             warped_img)
        fromNewWindows=True
        sanity_check_passed, l_fit_x_int, r_fit_x_int, x_int_diff, curvatures = run_sanity_check(img, left_poly_coeffs,
                                                                                     right_poly_coeffs, left_real_world,
                                                                                     right_real_world)
        if not sanity_check_passed:
            print("invalidating line fits")
            left_line.invalidate_last_fit()
            right_line.invalidate_last_fit()
            left_poly_coeffs = None
            right_poly_coeffs = None
    else:
        left_poly_coeffs, right_poly_coeffs, left_real_world, right_real_world = get_left_and_right_lane_lines_from_previous_fit(
            warped_img, left_line.best_fit, right_line.best_fit)
        fromNewWindows=False
        sanity_check_passed, l_fit_x_int, r_fit_x_int, x_int_diff, curvatures=run_sanity_check(img,left_poly_coeffs, right_poly_coeffs,left_real_world, right_real_world)
        if not sanity_check_passed:
            left_poly_coeffs, right_poly_coeffs, left_real_world, right_real_world, rectangles = get_left_and_right_lane_fits(img,
                                                                                                            warped_img)
            sanity_check_passed, l_fit_x_int, r_fit_x_int, x_int_diff, curvatures =run_sanity_check(img,left_poly_coeffs, right_poly_coeffs,left_real_world, right_real_world)
            if not sanity_check_passed:
                print("invalidating line fits")
                left_line.invalidate_last_fit()
                right_line.invalidate_last_fit()
                left_poly_coeffs = None
                right_poly_coeffs = None
            fromNewWindows = True
        if show_diagnostics:
            final_unwarped_img = draw_diagnostic_text_onto_image(img, l_fit_x_int, r_fit_x_int, x_int_diff,
                                                             img_index, fromNewWindows, curvatures)
    if left_poly_coeffs is not None and right_poly_coeffs is not None:
        left_path_coordinates, right_path_coordinates = get_left_and_right_lane_lines(left_poly_coeffs, right_poly_coeffs,img)
        y = left_path_coordinates[:, 1]
        y_eval = np.max(y)
        left_curvature, right_curvature = get_curvature(y_eval, left_real_world, right_real_world)
        left_line.radius_of_curvature = left_curvature
        right_line.radius_of_curvature = right_curvature
        avg_curvature = (left_curvature + right_curvature) / 2
        distance_from_lane_center = get_distance_from_lane_center(warped_img, left_path_coordinates, right_path_coordinates)
        unwarped_img = draw_lane_on_img(img, warped_img, left_path_coordinates, right_path_coordinates, MInv)
        final_unwarped_img = draw_text_onto_image(unwarped_img, avg_curvature, distance_from_lane_center)
        if show_diagnostics:
            final_unwarped_img = draw_diagnostic_text_onto_image(final_unwarped_img, l_fit_x_int, r_fit_x_int,x_int_diff,img_index,fromNewWindows,curvatures)
        process_lines(left_poly_coeffs, right_poly_coeffs,left_path_coordinates, right_path_coordinates ,avg_curvature,distance_from_lane_center)
    else:
        if show_diagnostics:
            final_unwarped_img = draw_diagnostic_text_onto_image(img, l_fit_x_int, r_fit_x_int, x_int_diff,
                                                             img_index, fromNewWindows, curvatures)
        else:
            final_unwarped_img=img
    return final_unwarped_img

def process_image(img):
    global num_times_called
    num_times_called=num_times_called+1
    if num_times_called==152:
        mpimg.imsave("test3.jpg",img)
    if num_times_called==184:
        mpimg.imsave("test4.jpg",img)
    final_unwarped_img=do_pipeline(img,num_times_called)
    return final_unwarped_img

def create_video(input_filename,output_filename):
    global num_times_called
    num_times_called=0
    left_line = Line()
    right_line = Line()
    clip = VideoFileClip(input_filename)
    processed_clip = clip.fl_image(process_image) #NOTE: this function expects color images!!
    processed_clip.write_videofile(output_filename, audio=False)

nx=9
ny=6
left_line = Line()
right_line = Line()
unwarped_imgs=[]
global num_times_called
num_times_called=0
args=get_arguments()
show_diagnostics=get_boolean_arg(args,"diagnostics")
do_visualization=get_boolean_arg(args,"visualization")
do_challenge_only=get_boolean_arg(args,"challenge")
do_basic_and_challenge=get_boolean_arg(args,"skipharderchallenge")

# Perform calibration using the 20 provided images in the camera_cal directory
ret, mtx, dist, rvecs, tvecs, orig_imgs, titles1, calib_imgs, titles2=calibrate_using_chessboard_images(nx,ny,'./camera_cal/calibration*.jpg')

if do_visualization:
    fig=show_array_2images_side_by_side(orig_imgs[11:14], titles1[11:14], calib_imgs[11:14], titles2[11:14],True)
    fig.savefig("./visualization/calibration_visualization.jpg",bbox_inches='tight')

    orig_imgs, titles1, undistorted_imgs, titles2=get_undistorted_images('./test_images/test*.jpg')
    fig = show_array_2images_side_by_side(orig_imgs[1:2], titles1[1:2], undistorted_imgs[1:2], titles2[1:2], True)
    fig.savefig("./visualization/undistort_visualization.jpg",bbox_inches='tight')

img=mpimg.imread('./test_images/straight_lines2.jpg')
M=setup_warp_transform(img)
MInv=setup_inverse_warp_transform(img)

if do_visualization:
    orig_imgs, titles1, warped_imgs, titles2=get_perspective_transform_images('./test_images/test*.jpg')
    fig = show_array_2images_side_by_side(orig_imgs, titles1, warped_imgs, titles2, True)
    fig.savefig("./visualization/perspective_transform_visualization.jpg",bbox_inches="tight")

if do_visualization:
    imgs,titles1,masked_imgs,titles2,warped_imgs,titles3=get_masked_warped_images('./test_images/test*.jpg')
    fig=show_array_3images_side_by_side(imgs, titles1, masked_imgs, titles2,warped_imgs,titles3,True)
    fig.savefig("./visualization/masked_warped.jpg",bbox_inches="tight")

    indx=2
    img=imgs[indx]
    warped_img = warped_imgs[indx]
    title=titles1[indx]
    visualize_search_windows(img,warped_img,title,"./visualization/search_windows1.jpg")
    indx = 3
    img = imgs[indx]
    warped_img = warped_imgs[indx]
    title = titles1[indx]
    visualize_search_windows(img, warped_img,title,"./visualization/search_windows2.jpg")

    for i, img in enumerate(warped_imgs):
        title=titles1[i]
        bottom_half=img[360:, :]
        histogram = get_xaxis_histogram(bottom_half)
        filename = title.split('/')[-1]
        view_histogram(histogram, filename, './visualization/histogram'+filename)

    for img_index, img in enumerate(orig_imgs):
        left_line = Line()
        right_line = Line()
        if img_index==5:
            print("index ={}".format(img_index))
        final_unwarped_img=do_pipeline(img,num_times_called)
        unwarped_imgs.append(final_unwarped_img)

    fig=show_array_2images_side_by_side(imgs, titles1, unwarped_imgs,titles2,True)
    fig.savefig("./visualization/showcompletepipeline.jpg",bbox_inches='tight')

if not do_visualization:
# Create the annotated videos
    if do_challenge_only:
        create_video('challenge_video.mp4', 'advanced_lane_finding_challenge.mp4')
    elif do_basic_and_challenge:
        create_video('project_video.mp4', 'advanced_lane_finding.mp4')
        create_video('challenge_video.mp4', 'advanced_lane_finding_challenge.mp4')
    else:
        create_video('project_video.mp4','advanced_lane_finding.mp4')
        create_video('challenge_video.mp4','advanced_lane_finding_challenge.mp4')
        create_video('harder_challenge_video.mp4','advanced_lane_finding_harder_challenge.mp4')
