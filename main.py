import math
import os
import threading

import cv2
import numpy as np
import pygame
import scipy
from scipy.spatial.transform import Rotation

import tracking_and_selection
import torch

from pygame import display, event, transform, mouse, draw, Rect, time, font

from depth_estimator import DepthEstimator
from util import mask_to_surface, ndarray_to_surface

device = torch.device('cuda')

VIDEO_WIDTH = 1280
VIDEO_HEIGHT = 720

screen = display.set_mode((VIDEO_WIDTH, VIDEO_HEIGHT))

cam = cv2.VideoCapture(0)

cam.set(cv2.CAP_PROP_FRAME_WIDTH, VIDEO_WIDTH)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, VIDEO_HEIGHT)
cam.set(cv2.CAP_PROP_FPS, 60)

target_box = None
running = True
dragging = False
click_start_pos = None
click_end_pos = None
clockity = time.Clock()
tracking_visualized = None
tracking_detections = None
depth_info = None
depth_visualized = None
frame = None
target_detection = None
target_mask = None
target_distance = None
fov_w = None
fov_h = None

def camera_task():
    global frame, running
    while running:
        ret, frame = cam.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

def calculate_fov(fov_x):
    fov_x_rad = 2 * math.atan2(VIDEO_WIDTH/2, fov_x)
    fov_y_rad = fov_x_rad * VIDEO_HEIGHT / VIDEO_WIDTH
    return fov_x_rad, fov_y_rad

def calculate_position(pixel_x, pixel_y, frame_w, frame_h, fov_w_rad, fov_h_rad, distance_m) -> (float, float, float):
    cx = frame_w / 2
    cy = frame_h / 2
    azimuth = ((pixel_x - cx) / frame_w) * fov_w_rad
    elevation = ((pixel_y - cy) / frame_h) * fov_h_rad
    forward_vector = np.array((0, 0, 1))
    aiming_vector = Rotation.from_euler('xy', (elevation, azimuth)).apply(forward_vector)

    return (distance_m / aiming_vector[2]) * aiming_vector

def tracking_task():
    global frame, target_box, tracking_visualized, running, tracking_detections, target_mask

    segmenter = tracking_and_selection.Tracker(device)
    print("Segmenter loaded")

    blob_detector_params = cv2.SimpleBlobDetector.Params()

    blob_detector_params.filterByCircularity = False
    blob_detector_params.filterByColor = False
    blob_detector_params.filterByConvexity = False
    blob_detector_params.filterByInertia = False
    blob_detector_params.filterByArea = True
    blob_detector_params.minArea = 1000
    blob_detector_params.maxArea = 1000000

    blob_detector = cv2.SimpleBlobDetector.create(blob_detector_params)
    logits = None
    masks = None
    try:
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            while running:
                if frame is None:
                    continue
                if target_box is not None or logits is not None:
                    if target_box is not None:
                        masks, scores, logits = segmenter.identify(frame, target_info=target_box)
                        target_box = None
                    elif logits is not None:
                        masks, scores, logits = segmenter.track(frame, previous_logits=logits)

                    sorted_ind = np.argsort(scores)[::-1]
                    masks = masks[sorted_ind]
                    scores = scores[sorted_ind]
                    logits = logits[sorted_ind]

                    target_mask = masks[0]

                    tracking_visualized = mask_to_surface(target_mask)
                    blob_mask = 255 * target_mask.astype(np.uint8)
                    tracking_detections = blob_detector.detect(blob_mask)
    except Exception as e:
        print("Segmenter failed:", e)

def depth_task():
    global frame, running, depth_info, depth_visualized, target_detection, target_mask, target_distance, fov_w, fov_h

    depth_estimator = DepthEstimator(device)
    print("Depth estimator loaded")

    MIN_DEPTH = 0.2
    MAX_DEPTH = 8
    with torch.inference_mode(), torch.autocast(device_type='cuda', dtype=torch.half):
        while running:
            if frame is None:
                continue
            depth_info, focal_length_x = depth_estimator.predict_depth(frame)
            fov_w, fov_h = calculate_fov(focal_length_x)
            if target_detection is not None and target_mask is not None:
                positions = np.indices((720, 1280), dtype=float)
                distances_from_target_x = np.power(positions[0] - target_detection.pt[0], 2)
                distances_from_target_y = np.power(positions[1] - target_detection.pt[1], 2)
                distances_from_target = distances_from_target_x + distances_from_target_y
                min_distance = np.min(distances_from_target)
                max_distance = np.max(distances_from_target)
                normalized_distances = (distances_from_target - min_distance) / (max_distance - min_distance)
                weight_mask = np.multiply(target_mask, 1 - normalized_distances)
                target_distance = np.multiply(depth_info, weight_mask).sum() / weight_mask.sum()

            clipped_depth = np.clip(depth_info, MIN_DEPTH, MAX_DEPTH)

            inverse_clipped = 1 / clipped_depth

            inverse_min_viz = min(inverse_clipped.max(), 1 / MIN_DEPTH)
            inverse_max_viz = max(inverse_clipped.min(), 1 / MAX_DEPTH)

            inverse_normalized = (inverse_clipped - inverse_max_viz) / (inverse_min_viz - inverse_max_viz)

            depth_image = (255 * inverse_normalized.clip(0, 1)).astype(np.uint8)
            surf_out = ndarray_to_surface(cv2.applyColorMap(depth_image, cv2.COLORMAP_OCEAN))
            surf_out.set_alpha(0.5*255)
            depth_visualized = surf_out

cam_thread = threading.Thread(target=camera_task, daemon=True)
segmentation_thread = threading.Thread(target=tracking_task, daemon=True)
depth_thread = threading.Thread(target=depth_task, daemon=True)
cam_thread.start()
depth_thread.start()
segmentation_thread.start()
font.init()
monocraft_font = font.SysFont('Monocraft', 20)

while running:
    for e in event.get():
        if e.type == pygame.QUIT:
            running = False
        elif e.type == pygame.MOUSEBUTTONDOWN:
            if e.button == pygame.BUTTON_LEFT:
                click_start_pos = e.pos
                dragging = True
        elif e.type == pygame.MOUSEBUTTONUP:
            if e.button == pygame.BUTTON_LEFT:
                click_end_pos = e.pos
                target_box = np.array((*click_start_pos, *click_end_pos))
                print(target_box)
                dragging = False
                depth_info = None

    if frame is not None:
        pyg_frame = transform.flip(transform.rotate(pygame.surfarray.make_surface(frame), -90), True, False)
        screen.blit(pyg_frame, (0, 0))

    if tracking_visualized is not None:
        screen.blit(tracking_visualized, (0, 0))
    if depth_visualized is not None:
        screen.blit(depth_visualized, (0, 0))

    if tracking_detections is not None:
        if tracking_detections:
            target_detection = max(tracking_detections, key=lambda detection: detection.size)
            draw.circle(screen, (0, 255, 0), target_detection.pt, target_detection.size / 2, 3)

            depth_area = Rect((0, 0), (target_detection.size/3, target_detection.size/3))
            depth_area.center = target_detection.pt

            draw.rect(screen, (0, 255, 0), depth_area, 2)

            if target_detection is not None and target_distance is not None and fov_w is not None and fov_h is not None:
                target_position_relative = calculate_position(target_detection.pt[0], target_detection.pt[1],
                                                              VIDEO_WIDTH, VIDEO_HEIGHT, fov_w, fov_h, target_distance)
                position_text = monocraft_font.render(f'Right/Left: {target_position_relative[0]:.2f} Up/Down: {target_position_relative[1]:.2f} Distance: {target_position_relative[2]:.2f}', True, (0, 255, 0))
                screen.blit(position_text, (target_detection.pt[0] + 10, target_detection.pt[1] + 10))
            elif target_distance is not None:
                distance_text = monocraft_font.render(f'{target_distance:.2f}', True, (0, 255, 0))
                screen.blit(distance_text, (target_detection.pt[0] + 10, target_detection.pt[1] + 10))
    mx, my = mouse.get_pos()

    draw.line(screen, (0, 100, 0), (VIDEO_WIDTH/2, 0), (VIDEO_WIDTH/2, VIDEO_HEIGHT))
    draw.line(screen, (0, 100, 0), (0, VIDEO_HEIGHT/2), (VIDEO_WIDTH, VIDEO_HEIGHT/2))

    if fov_w is not None:
        fov_w_text = monocraft_font.render(f"FOV_X {math.degrees(fov_w):.2f}", True, (0, 255, 0))
        screen.blit(fov_w_text, (5, 5))
    if fov_h is not None:
        fov_h_text = monocraft_font.render(f"FOV_Y {math.degrees(fov_h):.2f}", True, (0, 255, 0))
        screen.blit(fov_h_text, (5, 30))

    if fov_w is not None and fov_h is not None:
        fov_d_text = monocraft_font.render(f"FOV_D {math.hypot(math.degrees(fov_w), math.degrees(fov_h)):.2f}", True, (0, 255, 0))
        screen.blit(fov_d_text, (5, 55))

    if dragging:
        width = mx - click_start_pos[0]
        height = my - click_start_pos[1]
        target_rect = Rect(click_start_pos, (width, height))
        target_rect.normalize()
        draw.rect(screen, (0, 255, 0), target_rect, 4)

    if depth_info is not None:
        depth_text = monocraft_font.render(f'{depth_info[my, mx]:.2f}', True, (255, 255, 255))
        screen.blit(depth_text, (mx+10, my+10))

    display.set_caption(f'Position estimator')
    display.flip()
    clockity.tick(60)

pygame.quit()
if cam_thread.is_alive():
    cam_thread.join()
    cam.release()
if segmentation_thread.is_alive():
    segmentation_thread.join()
if depth_thread.is_alive():
    depth_thread.join()