from matplotlib import pyplot as plt
import cv2

def plot_image(image, title):
    img_rgb = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB) 
    plt.figure(figsize=(10,10))
    plt.imshow(img_rgb)
    plt.title(title)
    plt.show()