# %%
import numpy as np
import matplotlib.pyplot as plt

class image_unit:
    def __init__(self) -> None:
        self.num_of_points = 0
        self.cord_of_points = np.array([])
        self.raw_image = np.array([])
    def print_data(self) -> None:
        print(self.num_of_points)
        print(self.cord_of_points)
        print(self.raw_image)
        # Plotting the array as a grayscale image
        plt.imshow(self.raw_image, cmap='viridis')
        plt.colorbar()  # Adding a colorbar to show intensity scale
        plt.title('Visualization')
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.show()

# %%
# For every image:
# Line 1: number of points(10)
# Line 2-11: exact positions of the points
# Others: grayscale image
def read_in_data(file_path:str) -> list:
    with open(file_path, 'r') as file:
        _images_ = []
        cop = [] # Store cordinates of points tempoarily
        ri = [] # Store raw image tempoarily
        for line in file:
            numbers = []
            numbers = [float(s) for s in line.strip().split()]
            if len(numbers) == 1:
                if len(_images_) > 0:
                    _images_[-1].raw_image = np.array(ri).T
                    ri.clear()
                _images_.append(image_unit())
                _images_[-1].num_of_points = int(numbers[0])
            elif len(numbers) == 2:
                cop.append(numbers)
                if len(cop) == _images_[-1].num_of_points:
                    _images_[-1].cord_of_points = np.array(cop)
                    _images_[-1].cord_of_points = _images_[-1].cord_of_points
                    cop.clear()
            else:
                ri.append(numbers)
        _images_[-1].raw_image = np.array(ri).T
        return _images_

# %%
import cv2

# Define the color (B, G, R) and thickness of the cross marks
# color = (0, 255, 0)  # Green color
# cross_length = 10  # Length of the cross arms
# thickness = 2  # Thickness of the lines
def mark_point(image: np.array, points: list, color = (0, 255, 0), type = 'normal', cross_length = 2, thickness = 1) -> np.array:
    # Ensure the image is a NumPy array with the correct dtype
    if image.dtype != np.uint8:
        marked_image = (image / np.max(image) * 255).astype(np.uint8)
    else:
        marked_image = np.copy(image)
    if len(marked_image.shape) == 2 or (len(marked_image.shape) == 3 and marked_image.shape[2] == 1):
        marked_image = cv2.applyColorMap(marked_image, cv2.COLORMAP_TWILIGHT_SHIFTED)
    if type == 'normal':
        for (x, y) in points:
            cv2.line(marked_image, (x - cross_length, y), (x + cross_length, y), color, thickness)
            cv2.line(marked_image, (x, y - cross_length), (x, y + cross_length), color, thickness)
    elif type == 'skew':
        for (x, y) in points:
            cv2.line(marked_image, (x - cross_length, y - cross_length), (x + cross_length, y + cross_length), color, thickness)
            cv2.line(marked_image, (x + cross_length, y - cross_length), (x - cross_length, y + cross_length), color, thickness)
    # Display the image with marked points
    return marked_image

# %%
"""
    Extract a 5x5 subarray centered around the specified data point (center_x, center_y).
    
    Parameters:
    array (numpy.ndarray): The input array from which to extract the subarray.
    center_x (int): The x-coordinate (row) of the center data point.
    center_y (int): The y-coordinate (column) of the center data point.
    
    Returns:
    numpy.ndarray: The extracted 5x5 subarray, or None if the center point is too close to the border.
    """
def extract_subarr(array: np.array, center, half_size = 2): # A 5x5 array by default, half_size is 2
    (center_x, center_y) = center
    if (center_x < 0 or center_x >= array.shape[0] or center_y < 0 or center_y >= array.shape[1]):
        return None
    left = max(center_x - half_size, 0)
    left_pad = max(half_size - center_x, 0)
    right = min(center_x + half_size + 1, array.shape[0])
    right_pad = max(center_x + half_size + 1 - array.shape[0], 0)
    down = max(center_y - half_size, 0)
    down_pad = max(half_size - center_y, 0)
    up = min(center_y + half_size + 1, array.shape[1])
    up_pad = max(center_y + half_size + 1 - array.shape[1], 0)
    # Extract the 5x5 subarray centered around (center_x, center_y)
    subarray = array[down : up,
                     left : right]
    padded_arr = np.pad(subarray, ((down_pad, up_pad), (left_pad, right_pad)), mode='constant', constant_values=0)
    return padded_arr

def frac_part(array: np.array, center, half_size = 2):
    # print('array=', array) #debug
    # print('ccanter=', center) #debug
    sub_arr = extract_subarr(array, center, half_size).T
    # print('sub_arr=', sub_arr) #debug
    total_intensity = np.sum(sub_arr)
    result = np.array([- half_size + 0.5, - half_size + 0.5])
    if total_intensity == 0:
        return None
    for idx, x in np.ndenumerate(sub_arr):
        # print('idx = ', idx) #debug
        # print('np_arr', np.array(idx)) #debug
        # print('ratio', (x / total_intensity)) #debug
        result += np.array(idx) * (x / total_intensity)
    return result
    

# %%
import math
def distance(p, q):
    # Function to calculate Euclidean distance between points p and q
    return math.sqrt((q[0] - p[0])**2 + (q[1] - p[1])**2)

def smallest_distance_to_set(point, point_set):
    # point is a tuple (x, y) representing the point P
    # point_set is a list of tuples [(x1, y1), (x2, y2), ...] representing the set S
    
    if not point_set:
        return float('inf')  # If point_set is empty, return infinity
    
    x_distance = float('inf')
    y_distance = float('inf')
    min_distance = float('inf')
    
    for q in point_set:
        dist = distance(point, q)
        if dist < min_distance:
            min_distance = dist
            x_distance = point[0] - q[0]
            y_distance = point[1] - q[1]
    
    return [x_distance, y_distance, min_distance]

class judge_data:
    def __init__(self) -> None:
        self.dist = []
        self.rev_dist = []
        self.large_error = []
        self.large_error_points = []
        self.not_found = []
        self.not_found_points = []
        self.dist_array = np.array([])
    
    def print_data(self) -> None:
        # Testing code
        print(self.dist)  
        print(self.rev_dist)
        print(self.large_error)
        print(self.large_error_points)
        print(self.not_found)
        print(self.not_found_points)
        print(self.dist_array)
    
    def plot_histogram(self) -> None:
        # Plotting the histogram
        plt.hist(self.dist_array[0], bins=50, edgecolor='black')  # Adjust bins as needed
        plt.title('Histogram of Delta x')
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.show()

        plt.hist(self.dist_array[1], bins=50, edgecolor='black')  # Adjust bins as needed
        plt.title('Histogram of Delta y')
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.show()

        plt.hist(self.dist_array[2], bins=50, edgecolor='black')  # Adjust bins as needed
        plt.title('Histogram of Delta r')
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.show()
    
    def show_error(self, _images_: list, _found_: list, max_image_num: int = -1) -> None:
        if (len(self.large_error) == 0 and len(self.not_found) == 0):
            print('None')
        
        if max_image_num >= 0:
            image_num1 = min(len(self.large_error), max_image_num)
            image_num2 = min(len(self.not_found), max_image_num)
        else:
            image_num1 = len(self.large_error)
            image_num2 = len(self.not_found)
                             

        for i in range (image_num1):
            print('Image ' + str(self.large_error[i]) + ' Raw Data')
            print(extract_subarr(_images_[self.large_error[i]].raw_image, self.large_error_points[i]))
            marked_imag = mark_point(_images_[self.large_error[i]].raw_image, [tuple(np.round(row - 0.5).astype(int)) for row in _found_[self.large_error[i]]])
            marked_imag = mark_point(marked_imag, [tuple(row) for row in np.round(_images_[self.large_error[i]].cord_of_points - 0.5).astype(int)], (255, 0, 0))

            plt.imshow(marked_imag)
            plt.title('Image ' + str(self.large_error[i]) + ' Visualization')
            plt.xlabel('X-axis')
            plt.ylabel('Y-axis')
            plt.show()
        
        for i in range (image_num2):
            print('Image ' + str(self.not_found[i]) + ' Raw Data')
            print('Not Found:', self.not_found_points[i])
            print(extract_subarr(_images_[self.not_found[i]].raw_image, self.not_found_points[i]))
            marked_imag = mark_point(_images_[self.not_found[i]].raw_image, [tuple(row) for row in np.round(_images_[self.not_found[i]].cord_of_points - 0.5).astype(int)], (255, 0, 0))
            marked_imag = mark_point(marked_imag, [tuple(np.round(row - 0.5).astype(int)) for row in _found_[self.not_found[i]]])

            plt.imshow(marked_imag)
            plt.title('Image ' + str(self.not_found[i]) + ' Visualization')
            plt.xlabel('X-axis')
            plt.ylabel('Y-axis')
            plt.show()

def judge(_images_: list, _found_: list, threshold_error: float = 1.5) -> judge_data:
    _data_ = judge_data()
    cycles = min(len(_images_), len(_found_))

    for i in range (cycles):
        for j in range(len(_found_[i])):
            _data_.dist.append(smallest_distance_to_set(_found_[i][j], [tuple(row) for row in _images_[i].cord_of_points]))
            if _data_.dist[-1][2] > threshold_error:
                _data_.large_error.append(i)
                _data_.large_error_points.append(tuple(round(num - 0.5) for num in _found_[i][j]))
        for j in range(len(_images_[i].cord_of_points)):
            _data_.rev_dist.append(smallest_distance_to_set(_images_[i].cord_of_points[j], _found_[i]))
            if _data_.rev_dist[-1][2] > threshold_error:
                _data_.not_found.append(i)
                _data_.not_found_points.append(tuple(np.round(_images_[i].cord_of_points[j] - 0.5).astype(int)))

    _data_.dist_array = np.array(_data_.dist).T 

    return _data_
    

# %%
def find_local_maxima(matrix: np.array, threshold: int = 0):
    local_maxima = []
    for (i, j), value in np.ndenumerate(matrix):
        is_local_maxima = True
        if value < threshold: is_local_maxima = False
        # Check all eight possible neighbors
        neighbors = [
            (i-1, j-1), (i-1, j), (i-1, j+1),
            (i, j-1),             (i, j+1),
            (i+1, j-1), (i+1, j), (i+1, j+1)
        ]
        for u, v in neighbors:
            if 0 <= u < len(matrix) and 0 <= v < len(matrix[0]) and matrix[u][v] > value: 
                is_local_maxima = False
                break
        if is_local_maxima: local_maxima.append(np.array([j, i])) 
    return local_maxima

def remove_isolated_pixels(image: np.array, threshold: int = 0):
    # Make a copy of the image to modify
    processed_image = image.copy()
    for (i, j), value in np.ndenumerate(image):
        if value <= threshold: break
        to_delete = True
        neighbors = [
            (i-1, j-1), (i-1, j), (i-1, j+1),
            (i, j-1),             (i, j+1),
            (i+1, j-1), (i+1, j), (i+1, j+1)
        ]
        for u, v in neighbors:
            if 0 <= u < image.shape[0] and 0 <= v < image.shape[1] and image[u][v] > threshold: 
                to_delete = False
                break
        if to_delete: processed_image[i, j] = threshold
    return processed_image
    
def get_float_result(_images_: list, _found_: list, half_size = 2, to_int: bool = False) -> list:
    if to_int:
        _float_result_ = [[0 for _ in range(len(row))] for row in _found_]
        for i in range(len(_found_)):
            for j in range(len(_found_[i])):
                _float_result_[i][j] = np.round(_found_[i][j] + frac_part(_images_[i], _found_[i][j], half_size) - 0.5).astype(int)
    else:
        _float_result_ = [[0.0 for _ in range(len(row))] for row in _found_]
        for i in range(len(_found_)):
            for j in range(len(_found_[i])):
                _float_result_[i][j] = _found_[i][j] + frac_part(_images_[i], _found_[i][j], half_size)
    return _float_result_

# %% [markdown]
# ### Simple Local Meximum Approach

# %%
import torch

def rgb_to_grayscale(rgb_tensor):
    """
    Convert an RGB tensor to a grayscale tensor.
    
    Parameters:
        rgb_tensor (torch.Tensor): The input RGB tensor with shape [C, H, W], where C=3 for RGB.
    
    Returns:
        torch.Tensor: The grayscale tensor with shape [1, H, W].
    """
    # Ensure the tensor is on the right device and has the correct shape
    if rgb_tensor.shape[0] != 3:
        raise ValueError("Input tensor must have 3 channels (RGB).")

    # Define the weights for RGB to grayscale conversion
    weights = torch.tensor([1/3, 1/3, 1/3], dtype=rgb_tensor.dtype, device=rgb_tensor.device)

    # Convert RGB to grayscale using tensor operations
    gray_tensor = torch.tensordot(rgb_tensor, weights, dims=([0], [0]))
    
    # Add batch dimension if needed
    if len(gray_tensor.shape) == 2:
        gray_tensor = gray_tensor.unsqueeze(0)
    
    return gray_tensor

def min_max_normalize(tensor, new_min=0.0, new_max=1.0):
    """
    Perform min-max normalization on a tensor.
    
    Parameters:
        tensor (torch.Tensor): The input tensor to normalize.
        new_min (float): The new minimum value of the normalized tensor.
        new_max (float): The new maximum value of the normalized tensor.
    
    Returns:
        torch.Tensor: The normalized tensor.
    """
    # Ensure tensor is of type float for normalization
    tensor = tensor.float()
    
    # Calculate min and max values
    tensor_min = tensor.min()
    tensor_max = tensor.max()
    
    # Avoid division by zero
    if tensor_max == tensor_min:
        return torch.full_like(tensor, new_min)
    
    # Apply min-max normalization formula
    normalized_tensor = (tensor - tensor_min) / (tensor_max - tensor_min)
    
    # Scale to the desired range [new_min, new_max]
    normalized_tensor = normalized_tensor * (new_max - new_min) + new_min
    
    return normalized_tensor

# %%
from torchsr.models import edsr
import torch.nn.functional as F

def get_high_resolution_image(_images_: list, _scale_: int = 2, zero_padding: int = 2, num_of_images: int = -1):
    if num_of_images < 0:
        num_of_images = len(_images_)
    hr_model = edsr(scale=_scale_, pretrained=True)
    high_resolution_images = []
    for i in range(num_of_images):
        print('Processing image ', i , '...')
        min_val = np.min(_images_[i])
        max_val = np.max(_images_[i])
        lr_t = min_max_normalize(torch.from_numpy(_images_[i]).unsqueeze(0).float()).repeat(3, 1, 1).unsqueeze(0)
        sr_t = hr_model(F.pad(lr_t, (zero_padding, zero_padding, zero_padding, zero_padding), mode='constant', value=0.0))
        margin = _scale_ * zero_padding
        sr = torch.mean(min_max_normalize(sr_t[:,:,margin:-margin,margin:-margin].squeeze(0), min_val, max_val), dim = 0).detach().cpu().numpy()
        high_resolution_images.append(sr)
    return high_resolution_images

# %%
def find_points(_images_: list, algorithm: str, half_size: int = 2, _threshold_: int = 1000, num_of_images: int = -1) -> list: 
    '''
    algorithms: 'local maxima', 'local maxima denoised', 'local maxima denoised double', 
    'hr local maxima denoised', 'hr local maxima denoised double'
    '''
    if num_of_images < 0:
        num_of_images = len(_images_)
    _found_ = []
    if algorithm == 'local maxima':
        for i in range (num_of_images):
            _found_.append(find_local_maxima(_images_[i].raw_image, _threshold_))
        _found_ = get_float_result([_image_.raw_image for _image_ in _images_], _found_, half_size)
    if algorithm == 'local maxima denoised':
        _denoised_ = [remove_isolated_pixels(_image_.raw_image) for _image_ in _images_]
        for i in range (num_of_images):
            _found_.append(find_local_maxima(_denoised_[i], _threshold_))
        _found_ = get_float_result(_denoised_, _found_, half_size)
    if algorithm == 'local maxima denoised double':
        _denoised_ = [remove_isolated_pixels(_image_.raw_image) for _image_ in _images_]
        for i in range (num_of_images):
            _found_.append(find_local_maxima(_denoised_[i], _threshold_))
        _mid_result_ = get_float_result(_denoised_, _found_, half_size, to_int=True)    
        _found_ = get_float_result(_denoised_, _mid_result_, half_size)
    if algorithm == 'hr local maxima denoised':
        _denoised_ = [remove_isolated_pixels(_image_.raw_image) for _image_ in _images_]
        _hr_images_ = get_high_resolution_image(_denoised_, 4, 2, num_of_images)
        for i in range (len(_hr_images_)):
            _found_.append(find_local_maxima(_hr_images_[i], _threshold_))
        _found_ = get_float_result(_hr_images_, _found_, 2 * half_size)
        _found_ = [[value / 4 for value in sublist] for sublist in _found_]
    if algorithm == 'hr local maxima denoised double':
        _denoised_ = [remove_isolated_pixels(_image_.raw_image) for _image_ in _images_]
        _hr_images_ = get_high_resolution_image(_denoised_, 4, 2, num_of_images)
        for i in range (len(_hr_images_)):
            _found_.append(find_local_maxima(_hr_images_[i], _threshold_))
        _mid_result_ = get_float_result(_hr_images_, _found_, half_size, to_int=True) 
        _found_ = get_float_result(_hr_images_, _mid_result_, 2 * half_size)
        _found_ = [[value / 4 for value in sublist] for sublist in _found_]
    return _found_

# %%