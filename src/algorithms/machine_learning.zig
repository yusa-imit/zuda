/// Machine Learning Algorithms
///
/// Classic machine learning algorithms for clustering, classification, and pattern recognition.
///
/// Categories:
/// - **Clustering**: Unsupervised grouping of data points
///   - K-Means: Partition-based clustering (O(nkd×iter))
///
/// Future additions:
/// - K-Nearest Neighbors (KNN) for classification
/// - Decision Trees (CART, ID3) for supervised learning
/// - Naive Bayes for probabilistic classification
/// - Principal Component Analysis (PCA) for dimensionality reduction
///
/// Use cases:
/// - Customer segmentation
/// - Anomaly detection
/// - Image compression
/// - Document clustering
/// - Pattern recognition

pub const kmeans = @import("machine_learning/kmeans.zig").kmeans;
pub const KMeansResult = @import("machine_learning/kmeans.zig").KMeansResult;
pub const KMeansOptions = @import("machine_learning/kmeans.zig").KMeansOptions;
