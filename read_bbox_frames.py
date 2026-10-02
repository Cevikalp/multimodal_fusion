from pathlib import Path
import os
import cv2
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from rosbags.highlevel import AnyReader
from ultralytics import YOLO
from myutils import find_target_person_bbox
#import global_config

# ============================================================
# CONFIG
# ============================================================

# Safe defaults if global_config is missing in your execution directory
class GlobalConfigDummy(object):
    SAVE_HZ = 2.0
    IMAGE_TOPICS = ["/windows_{:02d}/image/compressed".format(i) for i in range(1, 20)]
    BB_MARGIN = 100
    CROP_MARGIN = 50
    REV_TOPICS = ["/windows_{:02d}/image/compressed".format(i) for i in [2, 4, 5, 7, 8, 11, 12, 13]]
    RGBREV_TOPICS = ["/windows_{:02d}/image/compressed".format(i) for i in [9]]
    CAMERA_MATRIX = np.array([[684., 0., 512.], [0., 570., 288.], [0., 0., 1.]], dtype=np.float32)
global_config = GlobalConfigDummy()


BB_MARGIN = 100
CROP_MARGIN = 70
MARGIN = BB_MARGIN + CROP_MARGIN

CAMERA_MATRIX = np.array([
    [684., 0., 512.],
    [0., 570., 288.],
    [0.,   0.,   1.]
], dtype=np.float32)

DIST_COEFFS = np.zeros(5)

IMAGE_WIDTH = 1024
IMAGE_HEIGHT = 576

#IMAGE_WIDTH = 1024//2
#IMAGE_HEIGHT = 576//2


# Topics with inverted coordinates
REV_TOPICS = [
    "/windows_{:02d}/image/compressed".format(i)
    for i in [4, 5, 7, 8, 9, 11, 12, 13]
]

# RGB Camera inverted
RGBREV_TOPICS = [
    "/windows_{:02d}/image/compressed".format(i)
    for i in [9]
]

# Image Topics
IMAGE_TOPICS = [
    "/windows_{:02d}/image/compressed".format(i)
    for i in range(1, 20)
]

# ============================================================
# TF HELPERS
# ============================================================

def quat_to_rotation(q):

    return Rotation.from_quat([
        q.x,
        q.y,
        q.z,
        q.w
    ]).as_matrix()


def load_camera_tf(reader, camera_frame):

    tf_connections = [
        c for c in reader.connections
        if c.topic == "/tf"
    ]

    for connection, timestamp, rawdata in reader.messages(
            connections=tf_connections):

        msg = reader.deserialize(rawdata, connection.msgtype)

        for tf in msg.transforms:

            if tf.child_frame_id != camera_frame:
                continue

            t = tf.transform.translation
            q = tf.transform.rotation

            R_wc = quat_to_rotation(q)

            camera_pos = np.array([
                t.x,
                t.y,
                t.z
            ], dtype=np.float32).reshape(3, 1)

            R_cw = R_wc.T
            tvec = -R_cw @ camera_pos

            rvec_cw, _ = cv2.Rodrigues(R_cw)

            return R_cw, tvec, rvec_cw

    raise RuntimeError(
        f"Camera frame not found: {camera_frame}"
    )

def check_bbox_cache(bbox_cache, threshold=0.75):
    """
    Returns:
        pass_flag : 1 if ratio of flag==1 >= threshold else 0
        ratio     : fraction of flag==1 samples
    """

    total = len(bbox_cache)

    if total == 0:
        return 0, 0.0

    num_ones = sum(
        1 for item in bbox_cache
        if item["bbox"][0] == 1
    )

    ratio = num_ones / total

    pass_flag = 1 if ratio >= threshold else 0

    return pass_flag, ratio

# ============================================================
# PROJECT 3D TRACK TO BBOX
# ============================================================

def project_track_to_bbox(
        position,
        R_cw,
        tvec,
        rvec_cw,
        doRGBRev,
        doRev):

    pos_camera = R_cw @ position.reshape(3, 1) + tvec

    if pos_camera[2, 0] <= 0:
        return None

    imgpt, _ = cv2.projectPoints(
        position.reshape(1, 3),
        rvec_cw,
        tvec,
        CAMERA_MATRIX,
        DIST_COEFFS
    )

    u, v = imgpt.ravel().astype(int)

    MARGIN = BB_MARGIN + CROP_MARGIN

    cx = IMAGE_WIDTH // 2
    cy = IMAGE_HEIGHT // 2

    # Apply coordinate inversion if needed
    if doRev:
        u = int(2 * cx - u)
        v = int(2 * cy - v)

    # -------------------------------------------------
    # Reject detections close to image borders
    # -------------------------------------------------

    EDGE_THRESHOLD = 120
    camera_flag = 1

    if (
        u < EDGE_THRESHOLD or
        u > IMAGE_WIDTH - EDGE_THRESHOLD or
        v < EDGE_THRESHOLD or
        v > IMAGE_HEIGHT - EDGE_THRESHOLD
    ):
        camera_flag = 0

    x1 = max(0, u - MARGIN)
    y1 = max(0, v - MARGIN)

    x2 = min(IMAGE_WIDTH - 1,  u + MARGIN)
    y2 = min(IMAGE_HEIGHT - 1, v + MARGIN)
            
    
   
    return (camera_flag, x1, y1, x2, y2, u, v)

def project_track_to_bbox_prev(
        position,
        R_cw,
        tvec,
        rvec_cw,
        doRGBRev,
        doRev):

    pos_camera = R_cw @ position.reshape(3, 1) + tvec

    if pos_camera[2, 0] <= 0:
        return None

    imgpt, _ = cv2.projectPoints(
        position.reshape(1, 3),
        rvec_cw,
        tvec,
        CAMERA_MATRIX,
        DIST_COEFFS
    )
  
    u, v = imgpt.ravel().astype(int)
    MARGIN = BB_MARGIN + CROP_MARGIN
    cx = IMAGE_WIDTH//2
    cy = IMAGE_HEIGHT//2

        
    x1 = max(0, u - MARGIN)
    y1 = max(0, v - MARGIN)

    x2 = min(IMAGE_WIDTH, u + MARGIN + 1)
    y2 = min(IMAGE_HEIGHT, v + MARGIN  + 1)

    if doRev: 
        u = int(2 * cx - u)
        v = int(2 * cy - v)

        x1 = max(0, u - MARGIN)
        y1 = max(0, v - MARGIN)
        x2 = min(IMAGE_WIDTH, u + MARGIN + 1)
        y2 = min(IMAGE_HEIGHT, v + MARGIN + 1)

    return (x1, y1, x2, y2, u, v)


# ============================================================
# MAIN
# ============================================================

def extract_track_frames(
        bag_path,
        track_ids,
        start_times,
        end_times,
        yolo_model,
        output_dir,
        frame_dir):

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(frame_dir, exist_ok=True)

    with AnyReader([Path(bag_path)]) as reader:

        bag_start_time = reader.start_time * 1e-9
        bag_end_time = reader.end_time * 1e-9
        bag_duration = bag_end_time - bag_start_time

        frames_cameras = {}
        bboxes_cameras = {}
        im_times_cameras = {}

        for cnn in range(0, 19):
            cn = cnn + 1
            if cn==5:
                frames_cameras[cnn] = []
                bboxes_cameras[cnn] = []
                im_times_cameras[cnn] = []
                continue


            camera_frame = f"/sensor_windows_{cn:02d}"
            print(camera_frame)
            image_topic=f"/windows_{cn:02d}/image/compressed"

            doRev = image_topic in global_config.REV_TOPICS
            doRGBRev = image_topic in global_config.RGBREV_TOPICS

            frames_samples = {}
            bboxes_samples = {}
            im_times_samples = {}
                      

            for s in range(len(start_times)):
                start_time = start_times[s]
                end_time = end_times[s]
                track_id = track_ids[s]

                if start_time < bag_start_time and \
                end_time <= bag_duration + 1:

                    start_time = bag_start_time + start_time
                    end_time = bag_start_time + end_time

                start_ns = int(start_time * 1e9)
                end_ns = int(end_time * 1e9)

                print(f"Using interval {start_time:.3f} -> {end_time:.3f}")

                # --------------------------------------------------
                # CAMERA TF
                # --------------------------------------------------

                R_cw, tvec, rvec_cw = load_camera_tf(
                    reader,
                    camera_frame
                )

                # --------------------------------------------------
                # LOAD ALL BBOXES
                # --------------------------------------------------

                bbox_cache = []

                det_connections = [
                    c for c in reader.connections
                    #if c.topic == "/tracked_detections_fused"
                    if c.topic == "/tracked_detections_fused_corrected"
                ]

                print("Loading detections...")

                for connection, timestamp, rawdata in reader.messages(
                        connections=det_connections,
                        start=start_ns,
                        stop=end_ns):

                    msg = reader.deserialize(
                        rawdata,
                        connection.msgtype
                    )

                    channel_names = [
                        ch.name for ch in msg.channels
                    ]

                    if "tracking_id" not in channel_names:
                        continue

                    tid_idx = channel_names.index(
                        "tracking_id"
                    )

                    for i, pt in enumerate(msg.points):

                        tid = int(
                            msg.channels[tid_idx].values[i]
                        )

                        if int(tid) != int(track_id):
                            continue

                        pos = np.array([
                            pt.x,
                            pt.y,
                            pt.z
                        ], dtype=np.float32)

                        bbox = project_track_to_bbox(
                            pos,
                            R_cw,
                            tvec,
                            rvec_cw,
                            doRGBRev,
                            doRev
                        )

                        if bbox is None:
                            continue

                        bbox_cache.append({
                            "time": timestamp * 1e-9,
                            "bbox": bbox
                        })

                print(
                    f"Found {len(bbox_cache)} detections "
                    f"for track {track_id}"
                )

                visibility_flag, pass_ratio = check_bbox_cache(bbox_cache, threshold=0.75)
                if visibility_flag == 0:
                    print(f"Samples did not pass the visibility test with pass ratio = {pass_ratio:.2f}")
                    frames_samples [s] = []
                    bboxes_samples [s] = []
                    im_times_samples [s] = []
                    continue

                # --------------------------------------------------
                # READ IMAGES if visibility test is passed
                # --------------------------------------------------
                image_connections = [
                    c for c in reader.connections
                    if c.topic == image_topic
                ]

                frame_counter = 0
                im_counter = 0
                frames = {}
                bboxes = {}
                im_times = {}

                for connection, timestamp, rawdata in reader.messages(
                        connections=image_connections,
                        start=start_ns,
                        stop=end_ns):

                    msg = reader.deserialize(
                        rawdata,
                        connection.msgtype
                    )

                    # ==================================================
                    # <<< FRAME EXTRACTED HERE >>>
                    # ==================================================
                    try:
                        if not hasattr(msg, "data"):
                            print(f"Skipping non-image message: {type(msg)}")
                            continue

                        if msg.data is None or len(msg.data) == 0:
                            image = None
                        else:
                            image = cv2.imdecode(
                                np.frombuffer(msg.data, dtype=np.uint8),
                                cv2.IMREAD_COLOR
                            )
                            if image.shape[0] != 576 or image.shape[1] != 1024:
                                image = cv2.resize(
                                    image,
                                (1024, 576),  # (width, height)
                                interpolation=cv2.INTER_LINEAR
                                )

                            if image is None:
                                print("Failed to decode image")
                                continue

                    except Exception as e:
                        print(f"Image decode error: {e}")
                        continue
                   
                    if doRGBRev:
                      #  image[:] = cv2.flip(image, -1)
                        doRev = False
                    

                    out_file = os.path.join(
                                    output_dir,
                                    f"camera_{cn:02d}_track_id{tid:02d}_frame_{im_counter:06d}.jpg"
                                )
                    

                    if int(tid) == int(track_id):
                        im_counter += 1
                    

                    image_time = timestamp * 1e-9

                    if len(bbox_cache) == 0:
                        continue

                    nearest = min(
                        bbox_cache,
                        key=lambda x:
                        abs(x["time"] - image_time)
                    )

                    # Require close temporal match
                    if abs(nearest["time"] - image_time) > 0.10:
                        continue

                    flag, x1, y1, x2, y2, u, v = nearest["bbox"]
                    if yolo_model is not None:
                        with torch.no_grad():
                            yolo_results = yolo_model(image, verbose = False)
                        yolo_boxes = yolo_results[0].boxes.cpu().numpy()
                        target_boxes = torch.tensor(yolo_boxes.xywh)
                        target_classes = torch.tensor(yolo_boxes.cls)

                        mask = target_classes == 0
                        target_boxes = target_boxes[mask]
                        target_classes = target_classes[mask]
                        print(f"Found {len(target_boxes)} class-0 objects")   

                    big_box = [x1,y1,x2,y2]
                    target_person_bbox = find_target_person_bbox(image, big_box, target_boxes, yolo_model, u, v, overlap_threshold=0.5)         

                    frame_path = os.path.join(
                    frame_dir,
                    f"camera_{cn:02d}_sample_number_{s+1:02d}_frame_{frame_counter:06d}.jpg"
                    )

                    cv2.imwrite(frame_path, image)

                    cv2.rectangle(
                        image,
                        (x1, y1),
                        (x2, y2),
                        (0, 255, 0),
                        2
                    )

                    cv2.circle(
                        image,          # image
                        (int(u), int(v)), # center
                        4,              # radius
                        (0, 0, 255),    # red (BGR)
                        -1              # filled circle
                    )

                    if target_person_bbox != []:
                        x1, y1, x2, y2 = target_person_bbox

                        cv2.rectangle(
                                image,
                                (x1, y1),
                                (x2, y2),
                                (0, 255, 255),
                                2
                            )
                        cv2.putText(
                                    image,
                                    f"ID {track_id}",
                                    (x1, max(20, y1 - 10)),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.7,
                                    (0, 255, 0),
                                    2
                                )
                    else:
                        x1, y1, x2, y2 = [0.0, 0.0, 0.0, 0.0]

                    
                   
                    # ==================================================
                    # <<< FRAME SAVED HERE >>>
                    # ==================================================

                    out_file = os.path.join(
                    output_dir,
                    f"camera_{cn:02d}_sample_number_{s+1:02d}_frame_{frame_counter:06d}.jpg"
                    )
                                        
                    cv2.imwrite(out_file, image)

                    bbox = [x1, y1, x2, y2]
                    frames[frame_counter] = frame_path
                    bboxes[frame_counter] = bbox
                    im_times[frame_counter] = image_time


                    frame_counter += 1

                print(
                    f"Saved {frame_counter} annotated frames "
                    f"to {output_dir}"
                )
                frames_samples [s] = frames
                bboxes_samples [s] = bboxes
                im_times_samples [s] = im_times

            frames_cameras[cnn] =  frames_samples
            bboxes_cameras[cnn] = bboxes_samples
            im_times_cameras[cnn] = im_times_samples

    return frames_cameras, bboxes_cameras, im_times_cameras


# ============================================================
# EXAMPLE
# ============================================================

if __name__ == "__main__":

    frames, bboxes = extract_track_frames(
        bag_path="/mnt/data/IndividualCustomers_Hatshop_Dataset_Dec2025/Bags/2024_06_29_s6_merged_tracked.bag",
        track_ids=[0, 0, 0],
        start_times=[60, 65, 70],
        end_times=[65, 70,75],
        camera_frame="/sensor_windows_03",
        image_topic="/windows_03/image/compressed",
        output_dir="output_frames"
    )