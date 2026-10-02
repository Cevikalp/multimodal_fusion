import numpy as np
import torch



def overlap_ratio_small_box(box1, box2):
    """
    box = (x1, y1, x2, y2)

    Returns:
        overlap_exists : bool
        ratio          : intersection_area / smaller_box_area
    """

    x11, y11, x12, y12 = box1
    x21, y21, x22, y22 = box2

    # Intersection rectangle
    ix1 = max(x11, x21)
    iy1 = max(y11, y21)

    ix2 = min(x12, x22)
    iy2 = min(y12, y22)

    if ix2 <= ix1 or iy2 <= iy1:
        return False, 0.0

    intersection_area = (ix2 - ix1) * (iy2 - iy1)

    area1 = (x12 - x11) * (y12 - y11)
    area2 = (x22 - x21) * (y22 - y21)

    smaller_area = min(area1, area2)

    ratio = intersection_area / float(smaller_area)

    return True, ratio

def find_target_person_bbox(
        image,
        big_box,
        target_boxes,
        yolo_model,
        u,
        v,
        overlap_threshold=0.5):

    """
    big_box     : [x1,y1,x2,y2]
    target_boxes: YOLO xywh detections
    u,v         : projected person location

    Returns:
        person_box = [x1,y1,x2,y2] or None
    """

    candidate_boxes = []

    # --------------------------------------------------
    # First pass:
    # Keep only YOLO boxes overlapping big_box
    # --------------------------------------------------

    for target_box in target_boxes:

        x1_yolo = int(target_box[0]) - int(target_box[2]) // 2
        y1_yolo = int(target_box[1]) - int(target_box[3]) // 2

        x2_yolo = int(target_box[0]) + int(target_box[2]) // 2
        y2_yolo = int(target_box[1]) + int(target_box[3]) // 2

        small_box = [
            x1_yolo,
            y1_yolo,
            x2_yolo,
            y2_yolo
        ]

        flag_overlap, ratio = overlap_ratio_small_box(
            small_box,
            big_box
        )

        if flag_overlap and ratio > overlap_threshold:
            candidate_boxes.append(small_box)

    # --------------------------------------------------
    # One candidate
    # --------------------------------------------------

    if len(candidate_boxes) == 1:
        return candidate_boxes[0]

    # --------------------------------------------------
    # Multiple candidates
    # Select closest center to (u,v)
    # --------------------------------------------------

    if len(candidate_boxes) > 1:

        best_box = []
        best_dist = 1e20

        for box in candidate_boxes:

            cx = (box[0] + box[2]) / 2.0
            cy = (box[1] + box[3]) / 2.0

            dist = np.sqrt(
                (cx - u) ** 2 +
                (cy - v) ** 2
            )

            if dist < best_dist:
                best_dist = dist
                best_box = box

        return best_box

    # --------------------------------------------------
    # No candidate:
    # run YOLO again on cropped big box
    # --------------------------------------------------

    bx1, by1, bx2, by2 = big_box

    crop = image[by1:by2, bx1:bx2]

    if crop.size == 0:
        return []

    with torch.no_grad():
        results = yolo_model(crop, verbose=False)

    boxes = results[0].boxes

    if len(boxes) == 0:
        return []

    # keep only class 0
    cls = boxes.cls.cpu().numpy()
    xywh = boxes.xywh.cpu().numpy()

    person_boxes = []

    for box, c in zip(xywh, cls):

        if int(c) != 0:
            continue

        x1 = int(box[0] - box[2] / 2)
        y1 = int(box[1] - box[3] / 2)

        x2 = int(box[0] + box[2] / 2)
        y2 = int(box[1] + box[3] / 2)

        # convert crop coordinates back to image coordinates

        person_boxes.append([
            x1 + bx1,
            y1 + by1,
            x2 + bx1,
            y2 + by1
        ])

    if len(person_boxes) == 0:
        return []

    if len(person_boxes) == 1:
        return person_boxes[0]

    # multiple detections in crop:
    # choose closest center to (u,v)

    best_box = []
    best_dist = 1e20

    for box in person_boxes:

        cx = (box[0] + box[2]) / 2.0
        cy = (box[1] + box[3]) / 2.0

        dist = np.sqrt(
            (cx - u) ** 2 +
            (cy - v) ** 2
        )

        if dist < best_dist:
            best_dist = dist
            best_box = box

    return best_box