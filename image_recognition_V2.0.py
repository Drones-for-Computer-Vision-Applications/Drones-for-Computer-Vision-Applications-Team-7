import cv2
import numpy as np
import os

#current problems:
#cannot recognize lion
#cannot recognize gazelle
#detection is done in rgb not hsv

#input catpure folder location here
capture_folder = r'2025-08-14 flights\capture_14'
image_extensions = ('.jpg', '.jpeg', '.png')

#parameters: masking rgb
lower_green = np.array([20, 60, 20])
upper_green = np.array([90, 255, 90])

#cap14 values
lower_green_hsv = np.array([95, 37, 35])
upper_green_hsv = np.array([162, 266, 255])

#parameters: animal classification
blur_size = 13 #= 2n+1 size of gauss blur
minimun_animal_size = 1000 #minimal animal size in pixelarea
similarity_threshold = 1 #0 is similar, > 1 is very different

#global
total_animal_count = np.array([0, 0, 0, 0, 0, 0]) # rhinon, hippo, lion, elephant, gazelle, # zebra see definition in get_animal_name()
num_images_with_animals = 0

#animal definitions
rhino = np.array([3.29949078e-01, 8.08497063e-02, 2.01878033e-03, 1.10055658e-03, 1.63220678e-06, 2.94424791e-04, -1.64251168e-07])
hippo = np.array([ 2.78394998e-01, 4.44242923e-02, 3.70173686e-03, 1.42137131e-03,  3.08328152e-06, 2.33302265e-04, -1.05983521e-06])
lion = np.array([3.56274114e-01, 9.88602231e-02, 1.83552275e-03, 1.25197425e-03, 1.89765671e-06, 3.92450181e-04, 3.02939873e-08])
elephant = np.array([2.67163920e-01, 4.07043675e-02, 1.09227307e-03, 3.13753685e-04, 1.73584808e-07, 3.53640193e-05, -6.00386202e-08])
gazelle = np.array([2.44230562e-01, 2.31581605e-02, 5.33853062e-04, 8.63914661e-04, 5.86405320e-07, 1.31468249e-04, 1.86373327e-08])
zebra = np.array([3.22894799e-01, 7.36305699e-02, 3.52526167e-03, 1.52646460e-03, 3.44124604e-06, 3.33645716e-04, 8.34564793e-07])



def compare_shapes(hu1, hu2):
    # Logarithmic scaling for better comparison
    epsilon = 1e-10
    hu1 = -np.sign(hu1) * np.log10(np.abs(hu1) + epsilon)
    hu2 = -np.sign(hu2) * np.log10(np.abs(hu2) + epsilon)
    return np.sum(np.abs(hu1 - hu2))



def get_animal_name(found_animal):
    #calc all similarities
    similarity = np.array([
        compare_shapes(rhino,found_animal),
        compare_shapes(hippo,found_animal),
        compare_shapes(lion,found_animal),
        compare_shapes(elephant,found_animal),
        compare_shapes(gazelle,found_animal),
        compare_shapes(zebra,found_animal)
    ])

    #choose best simiarity
    animal_index = np.argmin(similarity)
    chosen_animal = np.zeros_like(total_animal_count)
    chosen_animal[animal_index] = 1
    if chosen_animal[0]: return ('rhino', chosen_animal)
    if chosen_animal[1]: return ('hippo', chosen_animal)
    if chosen_animal[2]: return ('lion', chosen_animal)
    if chosen_animal[3]: return ('elephant', chosen_animal)
    if chosen_animal[4]: return ('gazelle', chosen_animal)
    if chosen_animal[5]: return ('zebra', chosen_animal)
    return (None,np.array([0,0,0,0,0]))


def create_mask(img):
    # perform Range masking in hsv
    img_hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    img_hsv_blurred = cv2.GaussianBlur(img_hsv, (blur_size, blur_size), 0)

    hsv_mask = cv2.inRange(img_hsv_blurred,lower_green_hsv,upper_green_hsv)
    hsv_kernel = np.ones((7, 7), np.uint8)
    hsv_cleaned_mask = cv2.morphologyEx(hsv_mask, cv2.MORPH_OPEN, hsv_kernel)
    return hsv_cleaned_mask


def detect_animals(mask, img):
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Prepare black image and normal image for drawing contours 
    contour_img = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    contour_img_2 = img.copy()

    animal_count = np.zeros_like(total_animal_count)


    for i, contour in enumerate(contours):
        # Calculate properties for this contour
        perimeter = cv2.arcLength(contour, closed=True)
        area = cv2.contourArea(contour)
        

        if(area > minimun_animal_size): # avoid small false detections
            M = cv2.moments(contour)
            if M["m00"] != 0:  # Avoid division by zero
                cx = int(M["m10"] / M["m00"])  # Centroid X
                cy = int(M["m01"] / M["m00"])  # Centroid Y

                # Draw contours and centroid with a unique color
                color = tuple(np.random.randint(0, 255, 3).tolist()) 
                cv2.drawContours(contour_img, [contour], -1, color, 2)
                cv2.circle(contour_img, (cx,cy), 1, (color), thickness=2)
                
                cv2.drawContours(contour_img_2, [contour], -1, color, 2)
                cv2.circle(contour_img_2, (cx,cy), 1, color, thickness=2)
                
                #determine if animal with hu moments
                animal = cv2.HuMoments(M).flatten()
                print(f'blob {i}: {animal}')
                animal_name, animal_identifier = get_animal_name(animal)
                if not (animal_name == None):
                    print(f'animal {i}: {animal_name}, {animal}')
                    animal_count += animal_identifier
                    cv2.putText(contour_img, animal_name, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 2, cv2.LINE_AA, )
                    cv2.putText(contour_img_2, animal_name, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 2, cv2.LINE_AA, )

    return animal_count, contour_img, contour_img_2





# Get all files in the folder
try:
     files = os.listdir(capture_folder)
except FileNotFoundError:
    print(f"Error: Folder '{capture_folder}' not found.")
image_files = [f for f in files if f.lower().endswith(image_extensions)]
print(f"Found {len(image_files)} image(s) to search.")

#process images
for i, image_file in enumerate(image_files):

    # Read the image
    image_path = os.path.join(capture_folder, image_file)
    img = cv2.imread(image_path)
    mask = create_mask(img)
    animal_count, marked_mask, marked_img = detect_animals(mask, img)
    
    #counting
    total_animal_count += animal_count
    if (animal_count > 0).any():
        num_images_with_animals += 1
        print(f'---img_{i}: found {animal_count} animals---')

    #safe images with animals
    cv2.imwrite(f'{capture_folder}\\marked\\img-{i}_{np.sum(animal_count)}-animals.jpg', marked_img)

    #TEMPORARY TESTING
    # if i ==10:
    #     marked_mask = cv2.resize(marked_mask, (0,0), fx = 0.2, fy = 0.2)
    #     marked_img = cv2.resize(marked_img, (0,0), fx = 0.2, fy = 0.2)
    #     mask = cv2.resize(mask, (0,0), fx = 0.2, fy = 0.2)
    #     cv2.imshow('test0',mask)
    #     cv2.imshow('test1',marked_img)
    #     cv2.imshow('test2',marked_mask)

#output
print(f"Found {num_images_with_animals} image(s) with animals.")
print(f"Found {np.sum(total_animal_count)} animals.")
print(f"Found {np.sum(total_animal_count[0])} rhino.")
print(f"Found {np.sum(total_animal_count[1])} hippo.")
print(f"Found {np.sum(total_animal_count[2])} lion.")
print(f"Found {np.sum(total_animal_count[3])} elephant.")
print(f"Found {np.sum(total_animal_count[4])} gazelle.")
print(f"Found {np.sum(total_animal_count[5])} zebra.")

cv2.waitKey(0)
cv2.destroyAllWindows