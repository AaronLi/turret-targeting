import numpy as np
import pygame
from pygame import transform, Surface


def mask_to_surface(mask):
    color = np.array([30, 144, 255])
    h, w = mask.shape[-2:]
    mask = mask.astype(np.uint8)
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    out_surface = ndarray_to_surface(mask_image)
    out_surface.set_alpha(255 * 0.6)
    return out_surface

def ndarray_to_surface(array) -> Surface:
    return transform.flip(transform.rotate(pygame.surfarray.make_surface(array), -90), True, False)