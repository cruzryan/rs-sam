import numpy as np
import pyrealsense2 as rs
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import open3d as o3d
import cv2

# Realsense setup (im using D415)

pipeline = rs.pipeline()
config = rs.config()

pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()

config.enable_stream(rs.stream.depth, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

pipeline.start(config)
profile = pipeline.get_active_profile()
depth_profile = rs.video_stream_profile(profile.get_stream(rs.stream.depth))
depth_intrinsics = depth_profile.get_intrinsics()
w, h = depth_intrinsics.width, depth_intrinsics.height

depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()

sensor = pipeline.get_active_profile().get_device().query_sensors()[1]

# Change exposure bc realsense suckss at automatically changing it
sensor.set_option(rs.option.exposure, 356.000)

print("Starting RS with depth_scale: ", depth_scale)

def get_points(display: bool) -> np.ndarray:
    pc = rs.pointcloud()
    decimate = rs.decimation_filter()
    decimate.set_option(rs.option.filter_magnitude, 2 ** 1)

    align_to = rs.stream.color
    align = rs.align(align_to)

    frames = pipeline.wait_for_frames()

    aligned_frames = align.process(frames)
    aligned_depth_frame = aligned_frames.get_depth_frame()
    color_frame = aligned_frames.get_color_frame()

    depth_image = np.asanyarray(aligned_depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())
    color_image_bgr = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)

    # For easy debugging
    depth_frame = frames.get_depth_frame()
    # depth_image = np.asanyarray(depth_frame.get_data())

    if display:
        plt.imshow(color_image_bgr)
        plt.title('Color Image')
        plt.pause(0.01) 

    plt.imsave('./example/test_no_prompt.jpg', color_image_bgr, cmap='gray')

    points = pc.calculate(depth_frame)

    v = points.get_vertices()
    verts = np.asanyarray(v).view(np.float32).reshape(-1, 3)  # xyz
    verts *= 1

    return verts 

def capture(display: bool) -> None:

    print("Capturing data...")

    verts = get_points(display)

    axes_lines = [[0, 0, 0], [5, 0, 0],  # x-axis
                  [0, 0, 0], [0, 5, 0],  # y-axis
                  [0, 0, 0], [0, 0, 5]]  # z-axis

    lines = [[0, 1], [2, 3], [4, 5]]  # line segments for x, y, z axes

    # Create Open3D geometries
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(verts)

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(axes_lines)
    line_set.lines = o3d.utility.Vector2iVector(lines)

    # Set colors for axes lines
    colors = [[1, 0, 0] for _ in range(len(lines))]  # red color for axes lines
    line_set.colors = o3d.utility.Vector3dVector(colors)

    # Saving pointcloud for later use
    o3d.io.write_point_cloud("./example/test.pcd", point_cloud)

    if display:
        # Visualize the point cloud and axes lines
        o3d.visualization.draw_geometries([point_cloud, line_set], window_name='3D Point Cloud', width=800, height=600)

    print(verts)