function probabilities = mixture_sampling(points, center_indices, alpha, distances_squared)
    m = length(points);
    total_distances_squared = sum(distances_squared);
    
    probabilities = ((1 - alpha) * distances_squared / total_distances_squared) + (alpha / m);
end