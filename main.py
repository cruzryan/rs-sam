import capture_data
import process_data
import numpy as np

if __name__ == "__main__":

    print("Starting...")

    ##### Uncomment this if you want to capture new data ######
    #capture_data.capture(display=True) 

    image_path = "./example/test_no_prompt.jpg"
    pointcloud_path = "./example/test.pcd"
    object_to_grab = "banana"

    process_data.process(image_path, pointcloud_path, object_to_grab, display=True)
