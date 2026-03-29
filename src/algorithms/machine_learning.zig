/// Machine Learning Algorithms
///
/// Classic machine learning algorithms for clustering, classification, and pattern recognition.
///
/// Categories:
/// - **Clustering** (Unsupervised): Group similar data points
///   - K-Means: Partition-based clustering (O(nkd×iter))
///
/// - **Classification** (Supervised): Predict categorical labels
///   - K-Nearest Neighbors (KNN): Instance-based learning (O(nd) per query)
///   - Decision Trees: Recursive partitioning with multiple criteria (O(nm log n) training)
///
/// Future additions:
/// - Naive Bayes for probabilistic classification
/// - Principal Component Analysis (PCA) for dimensionality reduction
/// - Support Vector Machines (SVM) for binary classification
///
/// Use cases:
/// - Customer segmentation (K-Means)
/// - Pattern recognition (KNN)
/// - Anomaly detection (K-Means, KNN)
/// - Image compression (K-Means)
/// - Medical diagnosis (KNN)
/// - Recommendation systems (KNN)

// Clustering algorithms
pub const kmeans = @import("machine_learning/kmeans.zig").kmeans;
pub const KMeansResult = @import("machine_learning/kmeans.zig").KMeansResult;
pub const KMeansOptions = @import("machine_learning/kmeans.zig").KMeansOptions;

// Classification algorithms
pub const KNN = @import("machine_learning/knn.zig").KNN;
pub const DecisionTree = @import("machine_learning/decision_tree.zig").DecisionTree;
