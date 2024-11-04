function [center_indices, cluster_number] = kmeans_with_mixture(x, k, delta, alpha)
    [m, ~] = size(x);
    center_indices = randi([1, m]); % Choosing first center randomly
    cluster_number = repmat(center_indices, [m, 1]);
   
    distances =pdist2(x,x(center_indices,:));
    
    for i = 1:(k-1) % Choosing 1/delta centers from mixture distribution in each iteration
        probabilities = mixture_sampling(x, center_indices, alpha, distances);
        a = randsample(1:m, floor(1/delta), true, probabilities);
        center_indices = [center_indices, a];

        % Compute pairwise distances between points and new centers
        dist_to_new_centers = pdist2(x, x(a, :));
        [min_d,~]=min(dist_to_new_centers,[],2);

       
        update_indices = min_d <= distances;

        distances(update_indices) = min_d(update_indices);
    end
end