"""

Helper functions for masks preprocessing

"""
import math

import cv2
import numpy as np

import rasterio
import rasterio.features


def get_polygons(mask, transform):
    polygons = []
    shapes = rasterio.features.shapes(mask, mask=mask == 1, transform=transform)
    shapes = filter(lambda ab: ab[1] == 1, shapes)
    shapes = list(shapes)

    if len(shapes) > 0:
        polygons = [polygon for (polygon, _) in shapes]
    else:
        polygons = []

    return polygons


def mask_denoise(mask, eps):
    """Removes noise from a mask.
    Args:
      mask: the mask to remove noise from.
      eps: the morphological operation's kernel size for noise removal, in pixel.
    Returns:
      The mask after applying denoising.
    Source: https://github.com/mapbox/robosat/blob/a8e0e3d676b454b61df03897e43e003867b6ef48/robosat/features/core.py  
    """
    struct = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (eps, eps))
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, struct)


"""
Get polygons from masks
Source: https://github.com/jamesmcclain/mask-to-polygons/blob/master/mask_to_polygons/processing/buildings.py
"""
def get_rectangle(buildings):
    _, contours, _ = cv2.findContours(buildings, cv2.RETR_EXTERNAL,
                                      cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 0:
        rectangle = cv2.minAreaRect(contours[0])
        return rectangle
    else:
        return None


def get_kernel(rectangle, width_factor=0.5, thickness=0.001):
    (_, (xwidth, ywidth), angle) = rectangle

    width = int(width_factor * min(xwidth, ywidth))
    half_width = width // 2
    diagonal = width * math.sqrt(2)
    pos = (half_width, half_width)
    dim = (diagonal, thickness)

    kernel = np.zeros((width, width), dtype=np.uint8)
    try:
        kernel = cv2.cvtColor(kernel, cv2.COLOR_GRAY2BGR)
    except Exception:
        return None

    if ywidth > xwidth:
        element = (pos, dim, angle)
    else:
        element = (pos, dim, angle + 90)
    element = cv2.boxPoints(element)
    element = np.int0(element)

    cv2.drawContours(kernel, [element], 0, (1, 0, 0), -1)
    kernel = kernel[:, :, 0]

    return kernel


def get_polygons(mask,
                 transform,
                 min_aspect_ratio=1.618,
                 min_area=None,
                 width_factor=0.5,
                 thickness=0.001):
    polygons = []
    if not min_area:
        min_area = mask.shape[0]  # Roughly the square root of the area
    n, components0 = cv2.connectedComponents(mask)

    for i in range(1, n):
        buildings = np.array((components0 == i), dtype=np.uint8)
        rectangle = get_rectangle(buildings)
        kernel = get_kernel(rectangle, width_factor, thickness)
        eroded = cv2.morphologyEx(
            buildings, cv2.MORPH_ERODE, kernel, iterations=1)
        m, components1 = cv2.connectedComponents(eroded)

        for j in range(1, m):
            building = np.array((components1 == j), dtype=np.uint8)
            if building.sum() >= min_area:
                dilated = cv2.morphologyEx(
                    building, cv2.MORPH_DILATE, kernel, iterations=1)
                shapes = rasterio.features.shapes(
                    dilated, transform=transform, mask=mask == 1)
                shapes = filter(lambda ab: ab[1] == 1, shapes)
                shapes = list(shapes)
                if len(shapes) > 0:
                    polygon, _ = shapes[0]
                    polygons.append(polygon)

    return polygons
