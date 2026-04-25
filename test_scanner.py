import cv2
import scanner

# Test with one image
image_path = "dataset/input_images/doc1.jpeg"
image = scanner.load_image(image_path)

if image is not None:
    print("Image loaded successfully")
    scanned, contour = scanner.process_image(image, bw_mode=False)
    if scanned is not None:
        print("Processing successful")
        cv2.imwrite("test_output.png", scanned)
        print("Output saved as test_output.png")
    else:
        print("Processing failed")
else:
    print("Failed to load image")