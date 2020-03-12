import genotype
import phenotype
import matplotlib.pyplot as plt
import numpy as np
from skimage.segmentation import find_boundaries


def simple_genetic_algorithm(
    image_path,
    population,
    episodes=100,
    mutation_rate=0.0001,
    crossover_rate=0.7,
    num_offsprings=15,
    p=0.2,
    weights=[0.1, 1.0, 3.0],
):
    weights = np.array(weights)
    population_size = len(population)

    def get_eval(segmentation):
        evaluation = phenotype.evaluate_segmentation(
            image, segmentation, min_region_size=1, max_regions=50
        )
        return np.sum(evaluation * weights)

    image = genotype.read_image(image_path, use_lab=False)
    population = np.array(population)
    segmentations = np.array(
        [phenotype.create_segmentation(image, i) for i in population]
    )
    fitness = np.array([get_eval(s) for s in segmentations])
    for generation in range(episodes):
        population, segmentations, fitness = sort_population(
            image, population, segmentations, fitness, get_eval, population_size
        )
        new_offsprings = []
        for _ in range(num_offsprings):
            parents = select(population, segmentations, fitness, population_size)
            offsprings = crossover(parents, crossover_rate=crossover_rate)
            offsprings = mutate(offsprings, mutation_rate=mutation_rate)

            for offspring in offsprings:
                segmentation = phenotype.create_segmentation(image, offspring)
                new_offsprings.append(offspring)
        population = np.append(population, new_offsprings, axis=0)
        segmentations = np.append(
            segmentations,
            [
                phenotype.create_segmentation(image, offspring)
                for offspring in new_offsprings
            ],
            axis=0,
        )
        fitness = np.append(fitness, [np.nan] * len(new_offsprings), axis=0)
        alpha = segmentations[0]
        print("Generation:", generation)
        print("Cost:", fitness[0])
        print("Number of segments:", alpha.max() + 1)
        print("-------------------------------------")
        segmentation = alpha.reshape(image.shape[:-1])
        type2 = find_boundaries(segmentation, mode="thick").astype(np.float)
        # Add borders
        type2[0, :] = 1
        type2[-1, :] = 1
        type2[:, 0] = 1
        type2[:, -1] = 1
        type1 = image.copy()
        type1[type2 == 1, 1] = 1
        plt.imsave(f"out/{generation}-1.png", type1, vmin=0, vmax=1)
        plt.imsave(f"out/{generation}-2.png", type2, cmap="Greys", vmin=0, vmax=1)


def sort_population(image, population, segmentations, fitness, eval_f, population_size):
    mask = np.isnan(fitness)
    fitness[mask] = [eval_f(s) for s in segmentations[mask]]

    sort_indices = fitness.argsort()
    population = population[sort_indices][:population_size]
    segmentations = segmentations[sort_indices][:population_size]
    fitness = fitness[sort_indices][:population_size]
    return population, segmentations, fitness


def select(population, segmentations, fitness, population_size, p=0.2):
    weights = np.array([1 * (1 - p) ** n for n in range(population_size)])
    weights = weights / np.linalg.norm(weights, ord=1)
    parents = np.random.choice(len(population), size=2, p=weights, replace=False)
    return [population[i].copy() for i in parents]


def crossover(parents, crossover_rate):
    genotype_length = parents[0].size
    c_mask_one = np.random.rand(genotype_length) < crossover_rate
    c_mask_two = np.random.rand(genotype_length) < crossover_rate
    offspring_one = parents[0].copy()
    offspring_two = parents[1].copy()
    offspring_one[c_mask_one] = parents[1][c_mask_one]
    offspring_two[c_mask_two] = parents[0][c_mask_two]
    return [offspring_one, offspring_two]


def mutate(offsprings, mutation_rate):
    genotype_length = offsprings[0].size
    for offspring in offsprings:
        mutation_mask = np.random.rand(genotype_length) < mutation_rate
        offspring[mutation_mask] = np.random.randint(0, 5, size=mutation_mask.sum())
    return offsprings


if __name__ == "__main__":
    image_path = "./train/86016/Test image.jpg"
    image = genotype.read_image(image_path)
    population = genotype.create_population(image_path, 50, use_lab=False)
    simple_genetic_algorithm(image_path, population)
