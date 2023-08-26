# Standard scientific Python imports
import matplotlib.pyplot as plt
import numpy as np
#classifiers and performance metrics
from sklearn import datasets, metrics
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
from scipy.io import loadmat
#load the dataset from a matlab file
caltech101 = loadmat('caltech101_silhouettes_28.mat')

X = caltech101['X']
Y = caltech101['Y']
class_names = caltech101['classnames']

Y = Y.flatten()
unique, counts = np.unique(Y, return_counts=True)
classes = np.isin(Y,unique[(counts > 80) & (unique != 2)])

# data contains the flattened images in a (samples, feature) matrix
data = X[classes]
#target 
target = Y[classes]
images = np.reshape(data,(len(target),28,28))
class_names_list = [class_names[0][i][0] for i in range(len(class_names[0]))]

#have a look at the shape of the data
print(data.shape)
print(target.shape)
print(images.shape)

# The data that we are interested in is made of 28x28 images from the caltech101
# dataset. Let's have a look at some images, stored in the `images` array.
# If we were working from image files, we could load them using matplotlib.pyplot.imread.
# Note that each image must have the same size. For these images, we know which class they represent:
# it is given in the 'target' of the dataset.
images_and_labels = list(zip(images, target))
plt.figure('Sample images')
for index, (image, label) in enumerate(images_and_labels[::len(target)//25]):
    if index == 25:
        break
    plt.subplot(5, 5, index+1)
    plt.tight_layout(pad=0.5)
    plt.axis('off')
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title(class_names_list[label-1],fontsize=10,color='r')
plt.show()

# we should 'scale' the data by ensuring zero mean and unit variance
data = scale(data.astype(float))

# Split the data into training and test sets 
X_train, X_test, y_train, y_test, images_train, images_test = train_test_split(data, target, images, test_size=0.25, random_state=42)

# create the MLP model
mlp = MLPClassifier()

# perform the training
mlp.fit(X_train, y_train)

# start evaluating
print("Training set score: %f" % mlp.score(X_train, y_train))
print("Test set score: %f" % mlp.score(X_test, y_test))

# Assign the predicted values to `predicted`
predicted = mlp.predict(X_test)

# Zip together the `images_test` and `predicted` values in `images_and_predictions`
images_and_predictions = list(zip(images_test, predicted))

plt.figure('Image predictions')
# Take a look at the first 16 predictions
for index, (image, prediction) in enumerate(images_and_predictions[::len(predicted)//25]):
    if index == 25:
        break
    # Initialize subplots in a grid of 5 by 5 at positions i+1
    plt.subplot(5, 5, index + 1)
    #Increase padding for titles
    plt.tight_layout(pad=1.0)
    # Don't show axes
    plt.axis('off')
    # Display images in all subplots in the grid
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    # Add a title to the plot
    plt.title(class_names_list[prediction-1],fontsize=10,color='r')

# Show the plot
plt.show()

# Print the classification report of `y_test` and `predicted`
print(metrics.classification_report(y_test, predicted))

# Print the confusion matrix
print(metrics.confusion_matrix(y_test, predicted))

