import numpy as np
from numba import jit
from skimage.segmentation import find_boundaries
from genotype import NEIGHBORS, _calculate_dist, _is_valid_vertex


def create_segmentation(image, genotype):
    return _create_phenotype(image, genotype).reshape(image.shape[:-1]).astype(np.int)


def evaluate_segmentation(
    image, segmentation, min_regions=1, max_regions=50, min_region_size=10
):
    unique, count = np.unique(segmentation, return_counts=True)
    if np.min(count) < min_region_size:
        return np.inf, np.inf, np.inf
    if not min_regions <= len(unique) < max_regions:
        return np.inf, np.inf, np.inf
    deviation_error = _deviation(image, segmentation)
    edge_error = _edge_value(image, segmentation)
    connectivity_error = _connectivity(image, segmentation)
    return deviation_error, edge_error, connectivity_error


def _edge_value(image, segmentation):
    boundary_diff = 0.0
    inner_boundary = find_boundaries(segmentation, mode="thick")
    for index in np.argwhere(inner_boundary):
        index = tuple(index)
        for n in index + NEIGHBORS[1:]:
            n = tuple(n)
            if _is_valid_vertex(image, n):
                if segmentation[index] != segmentation[n]:
                    boundary_diff += _calculate_dist(image, index, n)
    return -boundary_diff


def _deviation(image, segmentation):
    deviation = 0.0
    for segment in np.unique(segmentation):
        centroid = image[segmentation == segment].mean(axis=(0))
        flat_segment = image[segmentation == segment]
        deviation += np.sum(np.linalg.norm(flat_segment - centroid, axis=1))
    return deviation


def _connectivity(image, segmentation):
    outer_boundary = find_boundaries(segmentation, mode="thick")
    return np.sum(outer_boundary) * 1 / 8


@jit(nopython=True)
def _create_phenotype(image, genotype):
    segments = -np.ones(genotype.size)
    current_segment = 0
    for idx in range(segments.size):
        if segments[idx] < 0:
            search(idx, current_segment, genotype, segments, image.shape[1])
            current_segment += 1
    return segments


@jit(nopython=True)
def search(i, current_segment, genotype, segments, width):
    # Very simplified and ugly code to work with JIT
    # TODO: Cleanup
    segments[i] = current_segment
    j = i - 1
    u_direction = genotype[i]
    if i % width > 0 and segments[j] < 0:
        v_direction = genotype[j]
        if v_direction == 2 or u_direction == 1:
            search(j, current_segment, genotype, segments, width)
    j = i - width
    u_direction = genotype[i]
    if i - width >= 0 and segments[j] < 0:
        v_direction = genotype[j]
        if v_direction == 4 or u_direction == 3:
            search(j, current_segment, genotype, segments, width)
    j = i + 1
    u_direction = genotype[i]
    if j % width > 0 and segments[j] < 0:
        v_direction = genotype[j]
        if v_direction == 1 or u_direction == 2:
            search(j, current_segment, genotype, segments, width)
    j = i + width
    u_direction = genotype[i]
    if j < genotype.size and segments[j] < 0:
        v_direction = genotype[j]
        if v_direction == 3 or u_direction == 4:
            search(j, current_segment, genotype, segments, width)
