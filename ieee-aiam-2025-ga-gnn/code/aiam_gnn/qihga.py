"""Quantum-inspired mask search with bounded continuous genes and elitism."""

from dataclasses import dataclass

import numpy as np

from .graph_model import fit_candidate, probabilities


@dataclass
class Configuration:
    mask: np.ndarray
    hidden: int
    depth: int
    lr: float


@dataclass
class Chromosome:
    angles: np.ndarray
    # Width, depth and log10 learning rate are decoded from three unit-interval genes.
    hyper: np.ndarray

    def measure(self, rng, groups):
        probability = np.sin(self.angles) ** 2
        mask = rng.random(len(probability)) < probability
        for cols in groups:
            if not mask[cols].any():
                mask[cols[np.argmax(probability[cols])]] = True
        width = (8, 16, 32)[min(2, int(self.hyper[0] * 3))]
        depth = 1 + int(self.hyper[1] >= 0.5)
        return Configuration(mask, width, depth, 10 ** (-3.2 + 1.4 * self.hyper[2]))


def offspring(a, b, elite_mask, rng):
    weight = rng.random()
    # Interpolate alpha/beta and recover the normalized state angle.
    alpha = weight * np.cos(a.angles) + (1 - weight) * np.cos(b.angles)
    beta = weight * np.sin(a.angles) + (1 - weight) * np.sin(b.angles)
    angles = np.arctan2(beta, alpha)
    angles += np.where(elite_mask, 0.06, -0.06)
    mutate = rng.random(angles.size) < 0.10
    angles += mutate * rng.normal(0, 0.15, size=angles.size)
    blend = rng.random(3)
    hyper = blend * a.hyper + (1 - blend) * b.hyper + rng.normal(0, 0.10, size=3)
    return Chromosome(np.clip(angles, 0.05, np.pi / 2 - 0.05), np.clip(hyper, 0, 1))


def tournament(ranked, rng):
    """Choose the highest-ranked parent in a random draw of three."""
    positions = rng.integers(len(ranked), size=3)
    return ranked[min(positions)][1]


def search(train, labels, validation, val_labels, groups, args):
    rng = np.random.default_rng(args.seed)
    size = train.shape[1]
    population = [
        Chromosome(rng.uniform(0.55, 1.02, size), rng.random(3)) for _ in range(args.population)
    ]
    archive, trace = {}, []
    elite = None

    def assess(config, epochs, patience):
        model = fit_candidate(
            train,
            labels,
            validation,
            val_labels,
            groups,
            config,
            args.neighbors,
            args.encoder,
            epochs,
            patience,
            args.seed,
        )
        predicted = probabilities(model, validation).argmax(axis=1)
        accuracy = np.mean(predicted == val_labels)
        count = sum(p.numel() for p in model.parameters())
        # Bounded relative parameter-count proxy across masks and architectures.
        cost = count / (count + 1000)
        fitness = accuracy - 0.02 * config.mask.mean() - 0.01 * cost
        return float(fitness), model

    for generation in range(args.generations):
        # Keep the measured elite as well as its chromosome. Measuring it again
        # can draw a different mask and silently discard the winning phenotype.
        scored = [elite] if elite is not None else []
        for chromosome in population:
            config = chromosome.measure(rng, groups)
            key = (config.mask.tobytes(), config.hidden, config.depth, config.lr)
            if key not in archive:
                score, _ = assess(config, args.surrogate_epochs, args.surrogate_epochs)
                archive[key] = (score, config)
            score, config = archive[key]
            scored.append((score, chromosome, config))
        scored.sort(key=lambda item: item[0], reverse=True)
        elite = scored[0]
        trace.append(
            {
                "generation": generation + 1,
                "best_surrogate_fitness": elite[0],
                "unique_candidates_so_far": len(archive),
            }
        )

        population = [
            offspring(tournament(scored, rng), tournament(scored, rng), elite[2].mask, rng)
            for _ in range(args.population - 1)
        ]
    shortlist = sorted(archive.values(), key=lambda item: item[0], reverse=True)[: args.rescore]
    best_score = -np.inf
    for rank, (surrogate, config) in enumerate(shortlist, start=1):
        score, model = assess(config, args.epochs, args.patience)
        trace.append({"rescore_rank": rank, "surrogate_fitness": surrogate, "full_fitness": score})
        if score > best_score:
            best_score, best_config, best_model = score, config, model
    return best_config, best_model, trace
