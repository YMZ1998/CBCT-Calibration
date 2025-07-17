import cv2
from matplotlib import pyplot as plt


def plot(img, title, cmap=None):
    """Plot and show an image using matplotlib"""
    plt.figure(figsize=(9, 6))
    plt.imshow(img, cmap=cmap)
    plt.title(title)
    plt.xticks([]), plt.yticks([])
    plt.show()


def plotPoints(points, title):
    """Plot (x, y) points using scatter plot"""
    x = [p[0] for p in points]
    y = [p[1] for p in points]
    plt.figure(figsize=(9, 6))
    plt.title(title)
    plt.scatter(x, y)
    plt.show()


def plotParabolas(coefficients, x_vals, title, max_x, max_y):
    """Plot piecewise parabolic curves"""
    x, y, y_vals = [], [], []
    for i in range(len(x_vals) - 1):
        y_vals.append(
            coefficients[3 * i] * x_vals[i] ** 2 + coefficients[3 * i + 1] * x_vals[i] + coefficients[3 * i + 2])
        for x_val in range(x_vals[i], x_vals[i + 1]):
            x.append(x_val)
            y_val = coefficients[3 * i] * x_val ** 2 + coefficients[3 * i + 1] * x_val + coefficients[3 * i + 2]
            y.append(y_val)
    y_vals.append(
        coefficients[3 * i] * x_vals[-1] ** 2 + coefficients[3 * i + 1] * x_vals[-1] + coefficients[3 * i + 2])

    plt.figure(figsize=(9, 6))
    plt.title(title)
    plt.scatter(x, y)
    plt.plot(x_vals, y_vals, color='r')
    plt.plot(max_x, max_y, 'bo')
    plt.show()


def plotCubic(coefficients, x_vals, title, max_x, max_y):
    """Plot piecewise cubic splines"""
    x, y, y_vals = [], [], []
    for i in range(len(x_vals) - 1):
        y_vals.append(
            coefficients[4 * i] * x_vals[i] ** 3 +
            coefficients[4 * i + 1] * x_vals[i] ** 2 +
            coefficients[4 * i + 2] * x_vals[i] +
            coefficients[4 * i + 3]
        )
        for x_val in range(x_vals[i], x_vals[i + 1]):
            x.append(x_val)
            y_val = (
                coefficients[4 * i] * x_val ** 3 +
                coefficients[4 * i + 1] * x_val ** 2 +
                coefficients[4 * i + 2] * x_val +
                coefficients[4 * i + 3]
            )
            y.append(y_val)
    y_vals.append(
        coefficients[4 * i] * x_vals[-1] ** 3 +
        coefficients[4 * i + 1] * x_vals[-1] ** 2 +
        coefficients[4 * i + 2] * x_vals[-1] +
        coefficients[4 * i + 3]
    )
    plt.figure(figsize=(9, 6))
    plt.title(title, fontsize=12)
    plt.xlabel('Focus Distance [mm]', fontsize=10)
    plt.ylabel('Relative Sharpness Value', fontsize=10)
    plt.tick_params(labelsize=5)
    plt.scatter(x, y)
    plt.plot(x_vals, y_vals, color='r')
    plt.plot(max_x, max_y, 'bo')
    plt.show()


def saveImageRGB(name, img):
    cv2.imwrite(name, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


def saveImageBW(name, img):
    cv2.imwrite(name, img)


def readImageRGB(name):
    return cv2.cvtColor(cv2.imread(name, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


def readImageBW(name):
    return cv2.imread(name, cv2.IMREAD_GRAYSCALE)
