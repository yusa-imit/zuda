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
///   - Naive Bayes: Probabilistic classification with independence assumptions (O(nm) training)
///   - Support Vector Machine (SVM): Maximum margin classification with SMO (O(n²×iter) training)
///   - Random Forest: Ensemble learning with bagging and random features (O(n_trees × nm log n) training)
///   - Gradient Boosting: Sequential ensemble learning with gradient descent (O(n_trees × nmd) training)
///   - Logistic Regression: Linear classification with sigmoid function (O(n_iter × nm) training)
///
/// Future additions:
/// - Principal Component Analysis (PCA) for dimensionality reduction
/// - Neural Networks for deep learning
/// - Adaptive Boosting (AdaBoost) for ensemble learning
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
pub const GaussianNaiveBayes = @import("machine_learning/naive_bayes.zig").GaussianNaiveBayes;
pub const SVM = @import("machine_learning/svm.zig").SVM;
pub const RandomForest = @import("machine_learning/random_forest.zig").RandomForest;
pub const GradientBoosting = @import("machine_learning/gradient_boosting.zig").GradientBoosting;
pub const LogisticRegression = @import("machine_learning/logistic_regression.zig").LogisticRegression;
