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
        plt.xlabel('Y-axis')
        plt.ylabel('X-axis')
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
                images[-1].raw_image = np.array(ri)
                ri.clear()
            images.append(image_unit())
            images[-1].num_of_points = int(numbers[0])
        elif len(numbers) == 2:
            cop.append(numbers)
            if len(cop) == images[-1].num_of_points:
                images[-1].cord_of_points = np.array(cop)
                cop.clear()
        else:
            ri.append(numbers)
    images[-1].raw_image = np.array(ri)

print(len(images))

# Print the list of numbers
for i in range(10):
    print('No.', i + 1)
    print(type(images[i].raw_image[0][0]))
    images[i].print_data()

# %%
# return a musk according to a seed
def region_grow(image, seeds, threshold = 0):
    rows, cols = image.shape
    mask = np.zeros((rows, cols), dtype=int)
    to_visit = seeds
    while to_visit:
        x, y = to_visit.pop(0)
        if mask[x, y] == 0:
            mask[x, y] = 1
            neighbors = [(x-1, y-1), (x-1, y), (x-1, y+1), (x, y-1), (x, y+1), (x+1, y-1), (x+1, y), (x+1, y+1)]
                
            for nx, ny in neighbors:
                if 0 <= nx < rows and 0 <= ny < cols:
                    if mask[nx, ny] == 0 and image[nx, ny] > threshold:
                        to_visit.append((nx, ny))
    return mask


# find out all the masks
def find_clusters(image, threshold = 0):
    seeds = []
    image_copy = image
    masks = []
    for (i, j), value in np.ndenumerate(image_copy):
        if value > threshold:
            seeds.append([i, j])
            masks.append(region_grow(image_copy, seeds, threshold))
            seeds.clear()
            image_copy[masks[-1] == 1] = 0
    return masks

# %%
masks = find_clusters(images[7].raw_image, 40)
for i in range(len(masks)):
    plt.imshow(masks[i], cmap='viridis')
    plt.colorbar()  # Adding a colorbar to show intensity scale
    plt.title('Visualization')
    plt.xlabel('Y-axis')
    plt.ylabel('X-axis')
    plt.show()


