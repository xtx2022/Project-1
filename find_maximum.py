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
# Specify the path to your text file
file_path = './Clusters_2D_100.txt'
images = []

# For every image:
# Line 1: number of points(10)
# Line 2-11: exact positions of the points
# Others: grayscale image
with open(file_path, 'r') as file:
    cop = [] # Store cordinates of points tempoarily
    ri = [] # Store raw image tempoarily
    for line in file:
        numbers = []
        numbers = [float(s) for s in line.strip().split()]
        if len(numbers) == 1:
            if len(images) > 0:
                images[-1].raw_image = np.array(ri).T
                ri.clear()
            images.append(image_unit())
            images[-1].num_of_points = int(numbers[0])
        elif len(numbers) == 2:
            cop.append(numbers)
            if len(cop) == images[-1].num_of_points:
                images[-1].cord_of_points = np.array(cop)
                images[-1].cord_of_points = images[-1].cord_of_points - 0.5 # -0.5, now the center is at (0, 0)
                cop.clear()
        else:
            ri.append(numbers)
    images[-1].raw_image = np.array(ri).T

print(len(images))

# Print the list of numbers
for i in range(1):
    print('No.', i + 1)
    print(type(images[i].raw_image[0][0]))
    images[i].print_data()

# %%
def find_local_maxima(matrix):
    rows = len(matrix)
    cols = len(matrix[0])
    
    def is_local_maxima(i, j):
        current = matrix[i][j]
        # Check all eight possible neighbors
        neighbors = [
            (i-1, j-1), (i-1, j), (i-1, j+1),
            (i, j-1),            (i, j+1),
            (i+1, j-1), (i+1, j), (i+1, j+1)
        ]
        for x, y in neighbors:
            if 0 <= x < rows and 0 <= y < cols:
                if matrix[x][y] >= current:
                    return False
        return True

    local_maxima = []
    for i in range(rows):
        for j in range(cols):
            if is_local_maxima(i, j):
                local_maxima.append((j, i))
    
    return local_maxima

# Find the local maximum
local_maxima = find_local_maxima(images[0].raw_image)


print("Local Maxima:", local_maxima) # 局部最大值的位置

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
def extract_subarray(array, center, half_size = 2): # A 5x5 array by default, half_size is 2
    """
    Extract a 5x5 subarray centered around the specified data point (center_x, center_y).
    
    Parameters:
    array (numpy.ndarray): The input array from which to extract the subarray.
    center_x (int): The x-coordinate (row) of the center data point.
    center_y (int): The y-coordinate (column) of the center data point.
    
    Returns:
    numpy.ndarray: The extracted 5x5 subarray, or None if the center point is too close to the border.
    """
    (center_x, center_y) = center
    # Ensure the center is not too close to the borders
    left = max(center_x - half_size, 0)
    right = min(center_x + half_size + 1, array.shape[0])
    down = max(center_y - half_size, 0)
    up = min(center_y + half_size + 1, array.shape[1])
    # Extract the 5x5 subarray centered around (center_x, center_y)
    subarray = array[down : up,
                     left : right]
    return subarray

# %%
print([tuple(row) for row in images[0].cord_of_points])
marked_test = mark_point(images[0].raw_image, find_local_maxima(images[0].raw_image))
plt.imshow(marked_test)
plt.title('Visualization')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.show()
marked_test = mark_point(marked_test, [tuple(row) for row in np.round(images[0].cord_of_points).astype(int)], (255, 0, 0))
plt.imshow(marked_test)
plt.title('Visualization')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.show()

arr_test = [1.0, 1.3, 1.6, 1.9, 2.2]
print('Test', np.round(arr_test).astype(int))

for i in range(len(images[0].cord_of_points)):
    print(images[0].cord_of_points[i])
    print(np.round(images[0].cord_of_points[i]).astype(int))
    print(extract_subarray(images[0].raw_image, np.round(images[0].cord_of_points[i]).astype(int)))

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

dist = []
rev_dist = []
large_error = []
large_error_points = []
not_found = []
not_found_points = []

for i in range (len(images)):
    finded = find_local_maxima(images[i].raw_image)
    for j in range(len(finded)):
        dist.append(smallest_distance_to_set(finded[j], [tuple(row) for row in images[i].cord_of_points]))
        if dist[-1][2] > 1.5:
            large_error.append(i)
            large_error_points.append(finded[j])
    for j in range(len(images[i].cord_of_points)):
        rev_dist.append(smallest_distance_to_set(images[i].cord_of_points[j], finded))
        if rev_dist[-1][2] > 1.5:
            not_found.append(i)
            not_found_points.append(tuple(np.round(images[i].cord_of_points[j]).astype(int)))


print(dist)
dist_array = np.array(dist).T   
print(dist_array.shape)
print(dist_array)
print(large_error)
print(large_error_points)
print(not_found)
print(not_found_points)

# Plotting the histogram
plt.hist(dist_array[0], bins=50, edgecolor='black')  # Adjust bins as needed
plt.title('Histogram of Delta x')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

plt.hist(dist_array[1], bins=50, edgecolor='black')  # Adjust bins as needed
plt.title('Histogram of Delta y')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

plt.hist(dist_array[2], bins=50, edgecolor='black')  # Adjust bins as needed
plt.title('Histogram of Delta r')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

# %%
print(large_error)
print(large_error_points)

for i in range (len(large_error)):
    print('Image ' + str(large_error[i]) + ' Raw Data')
    # finded_list = find_local_maxima(images[large_error[i]].raw_image)
    # for j in range(len(finded_list)):
    #     print(extract_subarray(images[large_error[i]].raw_image, finded_list[j]))
    print(extract_subarray(images[large_error[i]].raw_image, large_error_points[i]))
    marked_imag = mark_point(images[large_error[i]].raw_image, find_local_maxima(images[large_error[i]].raw_image))
    marked_imag = mark_point(marked_imag, [tuple(row) for row in np.round(images[large_error[i]].cord_of_points).astype(int)], (255, 0, 0))
    plt.imshow(marked_imag)
    plt.title('Image ' + str(large_error[i]) + ' Visualization')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.show()

for i in range (len(not_found)):
    print('Image ' + str(not_found[i]) + ' Raw Data')
    print('Not Found:', not_found_points[i])
    print(extract_subarray(images[not_found[i]].raw_image, not_found_points[i]))
    marked_imag = mark_point(images[not_found[i]].raw_image, [tuple(row) for row in np.round(images[not_found[i]].cord_of_points).astype(int)], (255, 0, 0))
    marked_imag = mark_point(marked_imag, find_local_maxima(images[not_found[i]].raw_image))

    plt.imshow(marked_imag)
    plt.title('Image ' + str(not_found[i]) + ' Visualization')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.show()

# %%



