import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

def main():
    # Load an image
    image_path = r'2025-08-14 flights\capture_14\animal_presets\full.png'  # Change this to your image path
    img = cv2.imread(image_path)
    if img is None:
        print("Error: Could not load image")
        return
    
    # Convert to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Create initial mask with default HSV range
    lower_hsv = np.array([0, 50, 50])
    upper_hsv = np.array([30, 255, 255])
    mask = cv2.inRange(hsv, lower_hsv, upper_hsv)
    result = cv2.bitwise_and(img, img, mask=mask)
    
    # Create figure and subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    plt.subplots_adjust(bottom=0.35)
    
    # Display original and masked images
    img_display = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result_display = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    
    img_plot = ax1.imshow(img_display)
    ax1.set_title('Original Image')
    result_plot = ax2.imshow(result_display)
    ax2.set_title('Masked Result')
    
    # Create slider axes
    axcolor = 'lightgoldenrodyellow'
    ax_h_low = plt.axes([0.2, 0.25, 0.6, 0.03], facecolor=axcolor)
    ax_h_high = plt.axes([0.2, 0.20, 0.6, 0.03], facecolor=axcolor)
    ax_s_low = plt.axes([0.2, 0.15, 0.6, 0.03], facecolor=axcolor)
    ax_s_high = plt.axes([0.2, 0.10, 0.6, 0.03], facecolor=axcolor)
    ax_v_low = plt.axes([0.2, 0.05, 0.6, 0.03], facecolor=axcolor)
    ax_v_high = plt.axes([0.2, 0.00, 0.6, 0.03], facecolor=axcolor)
    
    # Create sliders
    s_h_low = Slider(ax_h_low, 'Hue Low', 0, 179, valinit=0)
    s_h_high = Slider(ax_h_high, 'Hue High', 0, 179, valinit=30)
    s_s_low = Slider(ax_s_low, 'Sat Low', 0, 255, valinit=50)
    s_s_high = Slider(ax_s_high, 'Sat High', 0, 255, valinit=255)
    s_v_low = Slider(ax_v_low, 'Val Low', 0, 255, valinit=50)
    s_v_high = Slider(ax_v_high, 'Val High', 0, 255, valinit=255)
    
    def update(val):
        # Get current slider values
        h_low = int(s_h_low.val)
        h_high = int(s_h_high.val)
        s_low = int(s_s_low.val)
        s_high = int(s_s_high.val)
        v_low = int(s_v_low.val)
        v_high = int(s_v_high.val)
        
        # Update HSV range
        lower_hsv = np.array([h_low, s_low, v_low])
        upper_hsv = np.array([h_high, s_high, v_high])
        
        # Apply mask
        mask = cv2.inRange(hsv, lower_hsv, upper_hsv)
        result = cv2.bitwise_and(img, img, mask=mask)
        
        # Update display
        result_display = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        result_plot.set_data(result_display)
        fig.canvas.draw_idle()
    
    # Register update function
    s_h_low.on_changed(update)
    s_h_high.on_changed(update)
    s_s_low.on_changed(update)
    s_s_high.on_changed(update)
    s_v_low.on_changed(update)
    s_v_high.on_changed(update)
    
    plt.show()

if __name__ == '__main__':
    main()