import numpy as np
import pyrealsense2 as rs
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D
import open3d as o3d
import cv2

import os
os.environ["KERAS_BACKEND"] = "jax"
import keras
from keras import ops
import keras_cv
import meshio

from groundingdino.util.inference import Model as GroundingDINO

CONFIG_PATH = "GroundingDINO_SwinT_OGC.py"
WEIGHTS_PATH = "groundingdino_swint_ogc.pth"

grounding_dino = GroundingDINO(CONFIG_PATH, WEIGHTS_PATH)


def get_object_pixels() -> np.ndarray:
    image_path = "./cache/test_segmented.jpg"
    image = cv2.imread(image_path)

    height = image.shape[0]
    new_height = height - 485

    # Remove the bottom part of the image
    cropped_image = image[:new_height, :]
    resized_image = cv2.resize(cropped_image, (1280, 720))

    resized_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGBA)

    tolerance = 21

    object_pixels = []
    object_pixels_path = './cache/object_pixels.txt'

    with open(object_pixels_path, 'w') as file:
        for row in range(resized_image.shape[0]): 
            for col in range(resized_image.shape[1]):  
                pixel_value = resized_image[row, col]
                r = pixel_value[0]
                g = pixel_value[1]
                b = pixel_value[2] 

                if not ((abs(r-120) < tolerance) and (abs(g-187) < tolerance) and (abs(b-255) < tolerance)):
                    resized_image[row, col] = [255,255,255,0]
                else:
                    object_pixels.append([row, col])
                    file.write(f"{row},{col}\n")

    cv2.imwrite("./cache/test_segmented_transparent.png", resized_image)

    foreground = resized_image
    background = cv2.imread('./example/test_no_prompt.jpg')

    alpha = foreground[:, :, 3]
    mask = np.zeros_like(background[:, :, :3])
    for c in range(3):
        mask[:, :, c] = alpha / 255.0

    # Overlay the foreground on the background using the mask
    overlayed = (mask * foreground[:, :, :3] + (1.0 - mask) * background).astype(np.uint8)
    cv2.imwrite('./cache/overlayed.jpg', overlayed)

    return np.array(object_pixels)


def get_point_info(flattened_array, y, x):
    index = (y * 1280) + x 
    if index < 0 or index >= len(flattened_array):
        raise IndexError("Point coordinates are out of bounds.")
    return flattened_array[index]

#Helper functions thx to Keras Docs!
def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(
        pos_points[:, 0],
        pos_points[:, 1],
        color="green",
        marker="*",
        s=marker_size,
        edgecolor="white",
        linewidth=1.25,
    )
    ax.scatter(
        neg_points[:, 0],
        neg_points[:, 1],
        color="red",
        marker="*",
        s=marker_size,
        edgecolor="white",
        linewidth=1.25,
    )

def show_box(box, ax):
    box = box.reshape(-1)
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor="green", facecolor=(0, 0, 0, 0), lw=2))

def inference_resizing(image, pad=True):
    image = ops.cast(image, dtype="float32")
    old_h, old_w = image.shape[0], image.shape[1]
    scale = 1024 * 1.0 / max(old_h, old_w)
    new_h = old_h * scale
    new_w = old_w * scale
    preprocess_shape = int(new_h + 0.5), int(new_w + 0.5)

    image = ops.image.resize(image[None, ...], preprocess_shape)[0]

    if pad:
        pixel_mean = ops.array([123.675, 116.28, 103.53])
        pixel_std = ops.array([58.395, 57.12, 57.375])
        image = (image - pixel_mean) / pixel_std
        h, w = image.shape[0], image.shape[1]
        pad_h = 1024 - h
        pad_w = 1024 - w
        image = ops.pad(image, [(0, pad_h), (0, pad_w), (0, 0)])
 
        image = image * pixel_std + pixel_mean
    return image


def grounding(image_path: str, object_to_grab: str, display: bool) -> np.ndarray:
    print("Preforming DINO Grounding...")
    image = np.array(keras.utils.load_img(image_path))
    image = ops.convert_to_numpy(inference_resizing(image))

    boxes = grounding_dino.predict_with_caption(image.astype(np.uint8), object_to_grab)
    boxes = np.array(boxes[0].xyxy)

    if display:
        plt.figure()
        plt.imshow(image / 255.0)
        plt.pause(0.01) 

    print(image)

    plt.axis("off")
    plt.savefig('./cache/test_resized.jpg', dpi=300, bbox_inches='tight', pad_inches=0)

    for box in boxes:
        print("Adding a box!")
        x_min, y_min, x_max, y_max = box
        width = x_max - x_min
        height = y_max - y_min
        rect = patches.Rectangle((x_min, y_min), width, height, linewidth=2, edgecolor='r', facecolor='none')
        plt.gca().add_patch(rect)
    plt.savefig('./cache/test_with_boxes.jpg', dpi=300, bbox_inches='tight', pad_inches=0)

    if display:
        plt.show() 
        plt.pause(0.01) 


    return boxes

def SAM(boxes: np.ndarray, display: bool):
    print("Preforming SAM...")
    
    # speed optimization with float16 dtype
    keras.mixed_precision.set_global_policy("mixed_float16")

    model = keras_cv.models.SegmentAnythingModel.from_preset("sam_huge_sa1b")
    image = np.array(keras.utils.load_img("./cache/test_resized.jpg"))
    image = inference_resizing(image)


    outputs = model.predict(
    {
        "images": np.repeat(image[np.newaxis, ...], boxes.shape[0], axis=0),
        "boxes": boxes.reshape(-1, 1, 2, 2),
    },
    batch_size=1,
    )

    for mask in outputs["masks"]:
        mask = inference_resizing(mask[0][..., None], pad=False)[..., 0]
        mask = ops.convert_to_numpy(mask) > 0.0
        show_mask(mask, plt.gca())

        if display:
            show_box(boxes, plt.gca())

    plt.axis("off")
    plt.savefig('./cache/test_segmented.jpg', dpi=300, bbox_inches='tight', pad_inches=0)

    if display:
        plt.show()
        plt.pause(0.1)


def generate_mesh(pointcloud_path: str, display: bool):

    pixels = get_object_pixels()

    pcd = o3d.io.read_point_cloud(pointcloud_path)
    pc_array = np.asarray(pcd.points)
  
    
    # get relevant 3d points
    points_array = []

    for i in range(0, len(pixels), 1):
        pc = get_point_info(pc_array, pixels[i][0], pixels[i][1])
        if not(pc[0] == 0 and pc[1] == 0 and pc[2] == 0 ):
            points_array.append(pc)


    # print("points array: ", points_array)
    print("px len: ", len(pixels))
    print("pa len: ", len(points_array))

    if display:
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(points_array)
        point_cloud.paint_uniform_color([1.0, 0.0, 0.0]) 
        point_cloud.estimate_normals() 

        visualizer = o3d.visualization.Visualizer()
        visualizer.create_window()
        visualizer.add_geometry(point_cloud)
        visualizer.run()

    cells = [("triangle", np.array([[0, 1, 2], [0, 2, 3]]))]

    # Create the mesh object
    mesh = meshio.Mesh(points_array, cells)

    # Save the mesh to a file
    meshio.write("./cache/triangle_mesh.vtk", mesh)

def process(image_path: str, pointcloud_path: str, object_to_grab: str, display: bool) -> None:
    
    #Get bounding boxes off text with DINO
    boxes = grounding(image_path, object_to_grab, display)
    
    #Getting a mask
    SAM(boxes, display)
    
    #Aligning mask with original image
    generate_mesh(pointcloud_path, display)


  

    

    



