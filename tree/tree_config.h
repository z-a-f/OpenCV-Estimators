#ifndef TREE_CONFIG_H_
#define TREE_CONFIG_H_

#include <vector>

struct TreeConfig {
    // Maximum depth of the tree. If -1, use default.
    int max_depth;
    // Minimum samples in a split. If -1, use default.
    int min_samples_split;
    // Class weights. If empty, use default.
    std::vector<double> class_weights;

    // Pruning flag for the tree. 
    // If < 1, no pruning
    // Otherwise prune using K-fold CV, with k being the value.
    int prune = 1;
};

#endif  // TREE_CONFIG_H_
