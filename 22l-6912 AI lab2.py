import numpy as np
import matplotlib.pyplot as plt  
from sklearn.datasets import load_iris
'''# Creating a 1D array (like a list)  
arr = np.array([1, 20, 19, 21, 4])  
print(arr)
# Creating a 2D array (like a matrix)  
twoDArray = np.array([[1, 2, 3], [4, 5, 6]])   
print(twoDArray)
# slicing 
# Getting elements from index 1 to 3 (not including index 3)   
arr = arr[1:4]   
print(arr) 
arr = twoDArray[:2, :2]  
print(arr) 

# negative indexing 
arr1 = np.array([10, 20, 30, 40, 50]) 
Arr1 = arr1[-3:]  
print(Arr1)


#matplotlib

#Creating a Simple Line Plot: 

x = [1, 2, 3, 4, 5]  
y = [1, 4, 9, 16, 25]  
# Creating a line plot  
plt.plot(x, y) 
plt.title('Simple Line Plot')  
plt.xlabel('X-axis Label')  
plt.ylabel('Y-axis Label')  
plt.grid(True)  
# Display the plot  
plt.show() 

 #Creating a Bar Chart:  
categories = ['A', 'B', 'C', 'D']  
values = [5, 7, 3, 8]  
plt.bar(categories, values)  
plt.title('Bar Chart Example')  
plt.xlabel('Category')  
plt.ylabel('Value')  
plt.show() 

'''
'''
#Q1
group_A = [12, 15, 14, 13, 16, 18, 19, 15, 14, 20, 17, 14, 15,40,45,50,62] 
group_B = [12, 17, 15, 13, 19, 20, 21, 18, 17, 16, 15, 14, 16, 15] 

plt.boxplot(group_A)
plt.title('Box Plot for Group A')
plt.xlabel('Category')
plt.ylabel('Measurement Values')

# show plot

plt.show() 

# Box plot for Group B
plt.boxplot(group_B)
plt.title('Box Plot for Group B')
plt.xlabel('Category')
plt.ylabel('Measurement Values')
plt.show() 



# subPlot 1: Group A
x = np.arange(len(group_A))
y = np.array(group_A)
plt.subplot(1, 2, 1)
plt.plot(x, y)
plt.title('Box Plot for Group A')
plt.xlabel('Index')
plt.ylabel('Measurement Values')


# subPlot 2: Group B
x = np.arange(len(group_B))
y = np.array(group_B)
plt.subplot(1, 2, 2)
plt.plot(x, y)
plt.title('Box Plot for Group B')
plt.xlabel('Index')
plt.ylabel('Measurement Values')

# Display the plots
plt.show()


'''
'''
#Q2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

file = open("genome.txt", "r")
genome_sequence = list(file.read().strip())  # Convert text into a list of characters
file.close()

# length of the genome sequence
genome_length = len(genome_sequence)

# Step 3: Generate helix coordinates using parametric equations
t = np.linspace(0, 4 * np.pi, genome_length)  # Generate parameter t
x = np.cos(t)  # X-coordinates
y = np.sin(t)  # Y-coordinates
z = np.linspace(0, 5, genome_length)  # Z-coordinates for vertical spread

# Combine coordinates into a single array
coordinates = np.column_stack((x, y, z))

#  Assign colors to each nucleotide in the sequence
color_map = {'A': 'red', 'T': 'blue', 'G': 'green', 'C': 'yellow'}
colors = [color_map[base] for base in genome_sequence]

# Step 5: Create a 3D scatter plot for the helix
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# Scatter plot with color coding
ax.scatter(coordinates[:, 0], coordinates[:, 1], coordinates[:, 2], c=colors, s=50)

ax.set_title("3D Helix Structure")
ax.set_xlabel("X-axis")
ax.set_ylabel("Y-axis")
ax.set_zlabel("Z-axis")


plt.show()

'''
#q3

import requests
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO

# Step 1: Download the image and convert to NumPy array
image_url = "https://raw.githubusercontent.com/appbrewery/webdev/main/birthday-cake3.4.jpeg"  # Replace with your image URL
response = requests.get(image_url)
image = Image.open(BytesIO(response.content))
image_np = np.array(image)

# Step 2: Plot the original image
plt.figure(figsize=(10, 10))
plt.subplot(1, 3, 1)
plt.imshow(image_np)
plt.title('Original Image')
plt.axis('off')

# Step 3: Rotate the image by 90 degrees counterclockwise
rotated_image = np.rot90(image_np)

# Step 4: Flip the image horizontally
flipped_image = np.fliplr(image_np)

# Step 5: Plot the rotated image
plt.subplot(1, 3, 2)
plt.imshow(rotated_image)
plt.title('Rotated Image')
plt.axis('off')

# Step 6: Plot the flipped image
plt.subplot(1, 3, 3)
plt.imshow(flipped_image)
plt.title('Flipped Image')
plt.axis('off')


plt.tight_layout()
plt.show()
img_array = np.array(image)

# Apply the grayscale conversion formula
gray_img = np.dot(img_array[..., :3], [0.299, 0.587, 0.114])

# Plot the grayscale image
plt.imshow(gray_img, cmap='gray')
plt.axis('off')  # Turn off axis
plt.show()


#Q4

'''


# Load the Iris dataset
iris = load_iris()

# Accessing the features (data) using NumPy array
X = np.array(iris.data)  # Features (sepal length, sepal width, petal length, petal width)
# Accessing the target labels (species)
Y = np.array(iris.target)  # Target variable (species: 0 for setosa, 1 for versicolor, 2 for virginica)

# 1. Use NumPy to calculate mean, median, and standard deviation for each feature
mean_values = np.mean(X, axis=0)
median_values = np.median(X, axis=0)
std_values = np.std(X, axis=0)

# Print results
print("Mean values for each feature:", mean_values)
print("Median values for each feature:", median_values)
print("Standard deviation for each feature:", std_values)

# Find the minimum and maximum values for each feature
min_values = np.min(X, axis=0)
max_values = np.max(X, axis=0)

# Print min and max values
print("Minimum values for each feature:", min_values)
print("Maximum values for each feature:", max_values)

# 2. Extract only the sepal length and sepal width as a NumPy array
sepal_data = X[:, :2]  # Sepal length and sepal width

# 3. Visualizations using Matplotlib

# Scatter plot of sepal length vs sepal width
plt.figure(figsize=(8, 6))
plt.scatter(sepal_data[:, 0], sepal_data[:, 1], c=Y, cmap='viridis')
plt.title('Sepal Length vs Sepal Width')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Sepal Width (cm)')
plt.colorbar(label='Species')
plt.show()

# Histogram showing the distribution of sepal length
plt.figure(figsize=(8, 6))
plt.hist(X[:, 0], bins=20, color='skyblue', edgecolor='black')
plt.title('Distribution of Sepal Length')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Frequency')
plt.show()

# Line plot to visualize the relationship between petal length and petal width
plt.figure(figsize=(8, 6))
plt.plot(X[:, 2], X[:, 3], marker='o', linestyle='-', color='orange')
plt.title('Petal Length vs Petal Width')
plt.xlabel('Petal Length (cm)')
plt.ylabel('Petal Width (cm)')
plt.show()
'''