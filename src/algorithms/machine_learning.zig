/// Machine Learning Algorithms
///
/// Classic machine learning algorithms for clustering, classification, and pattern recognition.
///
/// Categories:
/// - **Clustering** (Unsupervised): Group similar data points
///   - K-Means: Partition-based clustering (O(nkd×iter))
///   - Gaussian Mixture Model (GMM): Soft clustering via EM algorithm (O(nkd²×iter))
///   - DBSCAN: Density-based clustering with noise detection (O(n²) naive, finds arbitrary shapes)
///
/// - **Classification** (Supervised): Predict categorical labels
///   - K-Nearest Neighbors (KNN): Instance-based learning (O(nd) per query)
///   - Decision Trees: Recursive partitioning with multiple criteria (O(nm log n) training)
///   - Naive Bayes: Probabilistic classification with independence assumptions (O(nm) training)
///   - Support Vector Machine (SVM): Maximum margin classification with SMO (O(n²×iter) training)
///   - Random Forest: Ensemble learning with bagging and random features (O(n_trees × nm log n) training)
///   - Gradient Boosting: Sequential ensemble learning with gradient descent (O(n_trees × nmd) training)
///   - Logistic Regression: Linear classification with sigmoid function (O(n_iter × nm) training)
///   - AdaBoost: Adaptive boosting with weighted weak learners (O(n_learners × nm log n) training)
///
/// - **Regression** (Supervised): Predict continuous values
///   - Linear Regression: OLS or gradient descent for continuous prediction (O(nm² + m³) OLS, O(n_iter × nm) GD)
///   - Polynomial Regression: Non-linear modeling via polynomial features (O(np² + p³) OLS where p = poly features)
///   - Ridge Regression: L2 regularized regression for handling multicollinearity (O(nm² + m³) training)
///   - Lasso Regression: L1 regularized regression for feature selection and sparsity (O(n_iter × nm) coordinate descent)
///   - Elastic Net Regression: Combined L1+L2 regularization for balanced sparsity and shrinkage (O(n_iter × nm) coordinate descent)
///
/// - **Dimensionality Reduction** (Unsupervised): Reduce feature space
///   - Principal Component Analysis (PCA): Linear projection onto maximum variance directions (O(nm²) via eigendecomposition)
///
/// Future additions:
/// - Neural Networks for deep learning
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
pub const gmm = @import("machine_learning/gmm.zig").gmm;
pub const GMMResult = @import("machine_learning/gmm.zig").GMMResult;
pub const GMMOptions = @import("machine_learning/gmm.zig").GMMOptions;
pub const dbscan = @import("machine_learning/dbscan.zig").dbscan;
pub const DBSCANResult = @import("machine_learning/dbscan.zig").DBSCANResult;
pub const DBSCANOptions = @import("machine_learning/dbscan.zig").DBSCANOptions;

// Classification algorithms
pub const KNN = @import("machine_learning/knn.zig").KNN;
pub const DecisionTree = @import("machine_learning/decision_tree.zig").DecisionTree;
pub const GaussianNaiveBayes = @import("machine_learning/naive_bayes.zig").GaussianNaiveBayes;
pub const SVM = @import("machine_learning/svm.zig").SVM;
pub const RandomForest = @import("machine_learning/random_forest.zig").RandomForest;
pub const GradientBoosting = @import("machine_learning/gradient_boosting.zig").GradientBoosting;
pub const LogisticRegression = @import("machine_learning/logistic_regression.zig").LogisticRegression;
pub const AdaBoost = @import("machine_learning/adaboost.zig").AdaBoost;

// Regression algorithms
pub const LinearRegression = @import("machine_learning/linear_regression.zig").LinearRegression;
pub const PolynomialRegression = @import("machine_learning/polynomial_regression.zig").PolynomialRegression;
pub const RidgeRegression = @import("machine_learning/ridge_regression.zig").RidgeRegression;
pub const LassoRegression = @import("machine_learning/lasso_regression.zig").LassoRegression;
pub const ElasticNetRegression = @import("machine_learning/elastic_net_regression.zig").ElasticNetRegression;

// Dimensionality reduction algorithms
pub const PCA = @import("machine_learning/pca.zig").PCA;
