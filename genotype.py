import numpy as np
from numba import jit
from skimage import io, color
import queue
from tqdm import tqdm


NEIGHBORS = np.array([(0, 0), (0, -1), (0, 1), (-1, 0), (1, 0)], dtype=np.int)


def _graph_to_genotype(image, graph):
    que = queue.PriorityQueue()
    genotype = np.zeros(len(graph.keys()), dtype=np.int)
    mst = []
    visited = set()

    starting_vertex_idx = np.random.randint(len(graph.keys()))
    starting_vertex = list(graph.keys())[starting_vertex_idx]
    visited.add(starting_vertex)

    for next_to, weight in graph[starting_vertex].items():
        que.put((weight, starting_vertex, next_to))

    while que.empty() is False:
        edge = que.get()
        weight, frm, to = edge

        if to in visited:
            continue

        visited.add(to)
        _, node_u, node_v = edge
        raveled_index = np.ravel_multi_index(node_v, image.shape[:-1])
        genotype[raveled_index] = calculate_direction(node_v, node_u)
        mst.append(edge[1:])

        for next_to, weight in graph[to].items():
            if next_to not in visited:
                que.put((weight, to, next_to))

    return genotype


def read_image(image_path, use_lab=False):
    image = io.imread(image_path)[..., 0:3]
    if use_lab:
        return color.rgb2lab(image)
    else:
        return image / 255


@jit(nopython=True)
def _calculate_dist(image, p1, p2):
    return np.linalg.norm(image[p1] - image[p2])


@jit(nopython=True)
def _is_valid_vertex(image, vertex):
    shape = image.shape[0:2]
    return 0 <= vertex[0] < shape[0] and 0 <= vertex[1] < shape[1]


@jit(nopython=True)
def calculate_direction(u, v):
    direction = np.array([v[i] - u[i] for i in range(2)])
    curr_direction = 0
    for i in range(NEIGHBORS.shape[0]):
        if np.all(NEIGHBORS[i] == direction):
            return curr_direction
        curr_direction += 1


def _image_to_graph(image):
    valid_vertices = np.argwhere(np.ones_like(image[..., 0]))
    graph = dict.fromkeys([tuple(vertex) for vertex in valid_vertices])
    # iterate over all vertices
    for vertex in valid_vertices:
        vertex = tuple(vertex)
        adj = {}
        # iterate over all neighbors
        for neighbor in vertex + NEIGHBORS[1:]:
            neighbor = tuple(neighbor)
            if _is_valid_vertex(image, neighbor):
                adj[neighbor] = _calculate_dist(image, vertex, neighbor)
        graph[vertex] = adj
    return graph


def create_population(image_path, population_size, use_lab=False):
    image = read_image(image_path, use_lab=use_lab)
    graph = _image_to_graph(image)
    return [_graph_to_genotype(image, graph) for _ in tqdm(range(population_size))]
