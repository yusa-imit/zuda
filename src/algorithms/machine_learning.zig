/// Machine Learning Algorithms
///
/// Classic machine learning algorithms for clustering, classification, and pattern recognition.
///
/// Categories:
/// - **Clustering** (Unsupervised): Group similar data points
///   - K-Means: Partition-based clustering (O(nkd×iter))
///   - K-Medoids (PAM): Medoid-based clustering robust to outliers (O(k(n-k)²×iter))
///   - Gaussian Mixture Model (GMM): Soft clustering via EM algorithm (O(nkd²×iter))
///   - DBSCAN: Density-based clustering with noise detection (O(n²) naive, finds arbitrary shapes)
///   - OPTICS: Ordering points to identify clustering structure (O(n²) naive, hierarchical density-based)
///   - Hierarchical Clustering: Agglomerative clustering with dendrogram (O(n³) naive, O(n² log n) optimized)
///   - Mean Shift: Mode-seeking density-based clustering (O(n²×iter×d), automatic K discovery)
///   - Spectral Clustering: Graph-based clustering using Laplacian eigendecomposition (O(n² × k + n³), handles non-convex clusters)
///   - Affinity Propagation: Message-passing clustering with automatic K discovery (O(n²×iter), preference-based granularity control)
///
/// - **Classification** (Supervised): Predict categorical labels
///   - K-Nearest Neighbors (KNN): Instance-based learning (O(nd) per query)
///   - Decision Trees: Recursive partitioning with multiple criteria (O(nm log n) training)
///   - Naive Bayes: Probabilistic classification with independence assumptions (O(nm) training)
///   - Gaussian Discriminant Analysis (LDA/QDA): Probabilistic classification with Gaussian class-conditional densities (O(nmd²) training)
///   - Support Vector Machine (SVM): Maximum margin classification with SMO (O(n²×iter) training)
///   - Random Forest: Ensemble learning with bagging and random features (O(n_trees × nm log n) training)
///   - Gradient Boosting: Sequential ensemble learning with gradient descent (O(n_trees × nmd) training)
///   - XGBoost: eXtreme gradient boosting with regularization and 2nd-order optimization (O(n_trees × nmd log n) training)
///   - LightGBM: Light gradient boosting with leaf-wise growth and histogram-based splits (O(n_trees × nm × num_leaves) training)
///   - CatBoost: Categorical boosting with ordered boosting and symmetric trees (O(n_trees × nmd) training)
///   - Logistic Regression: Linear classification with sigmoid function (O(n_iter × nm) training)
///   - AdaBoost: Adaptive boosting with weighted weak learners (O(n_learners × nm log n) training)
///   - Perceptron: Simple linear classifier with online learning (O(n_epochs × nm) training)
///   - Softmax Regression: Multinomial logistic regression for true multi-class classification (O(n_iter × nmk) training)
///
/// - **Regression** (Supervised): Predict continuous values
///   - Linear Regression: OLS or gradient descent for continuous prediction (O(nm² + m³) OLS, O(n_iter × nm) GD)
///   - Polynomial Regression: Non-linear modeling via polynomial features (O(np² + p³) OLS where p = poly features)
///   - Ridge Regression: L2 regularized regression for handling multicollinearity (O(nm² + m³) training)
///   - Bayesian Ridge Regression: Bayesian approach with automatic regularization tuning (O(n_iter × (nm² + m³)) training, O(nm) prediction with uncertainty)
///   - Lasso Regression: L1 regularized regression for feature selection and sparsity (O(n_iter × nm) coordinate descent)
///   - Elastic Net Regression: Combined L1+L2 regularization for balanced sparsity and shrinkage (O(n_iter × nm) coordinate descent)
///   - Support Vector Regression (SVR): Epsilon-insensitive loss with kernel trick (O(n²×iter) training, O(n_sv) prediction)
///   - Gaussian Process Regression: Bayesian non-parametric regression with uncertainty quantification (O(n³) training, O(n) per prediction)
///
/// - **Dimensionality Reduction** (Unsupervised): Reduce feature space
///   - Principal Component Analysis (PCA): Linear projection onto maximum variance directions (O(nm²) via eigendecomposition)
///   - t-SNE: Non-linear manifold learning for visualization (O(n²×iter) exact algorithm)
///   - Self-Organizing Map (SOM): Competitive learning on topological grid (O(n×iter×grid_size×d) training)
///
/// - **Anomaly Detection** (Unsupervised): Identify outliers
///   - Isolation Forest: Ensemble of isolation trees (O(n_trees × ψ log ψ) training where ψ = subsample size)
///
/// - **Neural Networks** (Deep Learning): Multi-layer function approximation
///   - Multi-Layer Perceptron (MLP): Feedforward neural network with backpropagation (O(epochs × batches × L × n_max² × batch) training)
///   - RBF Network: Radial basis function network for regression/classification (O(n_centers × n × d + n_centers³) training)
///
/// - **Sequence Modeling** (Temporal): Sequential pattern recognition
///   - Hidden Markov Model (HMM): Probabilistic model for sequences with hidden states (O(T × N²) forward/viterbi where T = sequence length, N = states)
///   - Conditional Random Field (CRF): Discriminative sequence labeling with feature functions (O(T × N² × K) training where K = features)
///
/// - **Reinforcement Learning** (Agent-Environment): Learning optimal policies through interaction
///   - Q-Learning: Off-policy TD learning for optimal action-value function (O(|A|) per update, O(|S| × |A|) space)
///   - SARSA: On-policy TD learning that learns policy being followed (O(|A|) per update, O(|S| × |A|) space)
///   - Expected SARSA: On-policy TD learning with expected value update (O(|A|) per update, O(|S| × |A|) space, lower variance than SARSA)
///   - Actor-Critic: Policy gradient with value baseline (O(|A|) per update, O(|S| + |S|×|A|) space, foundation for A2C/A3C/PPO)
///   - REINFORCE: Monte Carlo policy gradient (O(|A| × T) per episode, O(|S| × |A|) space, foundational policy gradient algorithm)
///   - DQN: Deep Q-Network with experience replay and target network (O(batch × network) per update, handles large state spaces)
///   - DDPG: Deep Deterministic Policy Gradient for continuous control (O(batch × network) per update, actor-critic with replay buffer)
///   - PPO: Proximal Policy Optimization with clipped objective (O(K × epochs × |A|) per update, stable on-policy learning)
///   - TD3: Twin Delayed DDPG with clipped double Q-learning (O(batch × network) per update, improved stability over DDPG)
///
/// Use cases:
/// - Customer segmentation (K-Means, K-Medoids)
/// - Pattern recognition (KNN)
/// - Anomaly detection (Isolation Forest, DBSCAN)
/// - Image compression (K-Means)
/// - Medical diagnosis (KNN)
/// - Recommendation systems (KNN)
/// - Fraud detection (Isolation Forest)
/// - Robust clustering with outliers (K-Medoids)
/// - Time series clustering with DTW (K-Medoids)

// Clustering algorithms
pub const kmeans = @import("machine_learning/kmeans.zig").kmeans;
pub const KMeansResult = @import("machine_learning/kmeans.zig").KMeansResult;
pub const KMeansOptions = @import("machine_learning/kmeans.zig").KMeansOptions;
pub const kmedoids = @import("machine_learning/kmedoids.zig").kmedoids;
pub const KMedoidsResult = @import("machine_learning/kmedoids.zig").KMedoidsResult;
pub const KMedoidsOptions = @import("machine_learning/kmedoids.zig").KMedoidsOptions;
pub const manhattanDistance = @import("machine_learning/kmedoids.zig").manhattanDistance;
pub const gmm = @import("machine_learning/gmm.zig").gmm;
pub const GMMResult = @import("machine_learning/gmm.zig").GMMResult;
pub const GMMOptions = @import("machine_learning/gmm.zig").GMMOptions;
pub const dbscan = @import("machine_learning/dbscan.zig").dbscan;
pub const DBSCANResult = @import("machine_learning/dbscan.zig").DBSCANResult;
pub const DBSCANOptions = @import("machine_learning/dbscan.zig").DBSCANOptions;
pub const OPTICS = @import("machine_learning/optics.zig").OPTICS;
pub const HierarchicalClustering = @import("machine_learning/hierarchical_clustering.zig").HierarchicalClustering;
pub const HierarchicalClusteringConfig = @import("machine_learning/hierarchical_clustering.zig").HierarchicalClusteringConfig;
pub const LinkageMethod = @import("machine_learning/hierarchical_clustering.zig").LinkageMethod;
pub const MergeStep = @import("machine_learning/hierarchical_clustering.zig").MergeStep;
pub const meanShift = @import("machine_learning/mean_shift.zig").meanShift;
pub const MeanShiftResult = @import("machine_learning/mean_shift.zig").MeanShiftResult;
pub const MeanShiftOptions = @import("machine_learning/mean_shift.zig").MeanShiftOptions;
pub const SpectralClustering = @import("machine_learning/spectral_clustering.zig").SpectralClustering;
pub const affinityPropagation = @import("machine_learning/affinity_propagation.zig").affinityPropagation;
pub const AffinityPropagationResult = @import("machine_learning/affinity_propagation.zig").AffinityPropagationResult;
pub const AffinityPropagationOptions = @import("machine_learning/affinity_propagation.zig").AffinityPropagationOptions;

// Classification algorithms
pub const KNN = @import("machine_learning/knn.zig").KNN;
pub const DecisionTree = @import("machine_learning/decision_tree.zig").DecisionTree;
pub const GaussianNaiveBayes = @import("machine_learning/naive_bayes.zig").GaussianNaiveBayes;
pub const GaussianDiscriminant = @import("machine_learning/gaussian_discriminant.zig").GaussianDiscriminant;
pub const SVM = @import("machine_learning/svm.zig").SVM;
pub const RandomForest = @import("machine_learning/random_forest.zig").RandomForest;
pub const GradientBoosting = @import("machine_learning/gradient_boosting.zig").GradientBoosting;
pub const XGBoost = @import("machine_learning/xgboost.zig").XGBoost;
pub const LightGBM = @import("machine_learning/lightgbm.zig").LightGBM;
pub const CatBoost = @import("machine_learning/catboost.zig").CatBoost;
pub const LogisticRegression = @import("machine_learning/logistic_regression.zig").LogisticRegression;
pub const AdaBoost = @import("machine_learning/adaboost.zig").AdaBoost;
pub const Perceptron = @import("machine_learning/perceptron.zig").Perceptron;
pub const SoftmaxRegression = @import("machine_learning/softmax_regression.zig").SoftmaxRegression;

// Regression algorithms
pub const LinearRegression = @import("machine_learning/linear_regression.zig").LinearRegression;
pub const PolynomialRegression = @import("machine_learning/polynomial_regression.zig").PolynomialRegression;
pub const RidgeRegression = @import("machine_learning/ridge_regression.zig").RidgeRegression;
pub const BayesianRidge = @import("machine_learning/bayesian_ridge.zig").BayesianRidge;
pub const LassoRegression = @import("machine_learning/lasso_regression.zig").LassoRegression;
pub const ElasticNetRegression = @import("machine_learning/elastic_net_regression.zig").ElasticNetRegression;
pub const SVR = @import("machine_learning/svr.zig").SVR;
pub const SVRKernelType = @import("machine_learning/svr.zig").KernelType;
pub const GaussianProcess = @import("machine_learning/gaussian_process.zig").GaussianProcess;
pub const GaussianProcessConfig = @import("machine_learning/gaussian_process.zig").Config;
pub const KernelType = @import("machine_learning/gaussian_process.zig").KernelType;

// Dimensionality reduction algorithms
pub const PCA = @import("machine_learning/pca.zig").PCA;
pub const TSNE = @import("machine_learning/tsne.zig").TSNE;
pub const TSNEOptions = @import("machine_learning/tsne.zig").TSNEOptions;
pub const SOM = @import("machine_learning/som.zig").SOM;

// Anomaly detection algorithms
pub const IsolationForest = @import("machine_learning/isolation_forest.zig").IsolationForest;

// Neural networks
pub const MLP = @import("machine_learning/mlp.zig").MLP;
pub const RBFNetwork = @import("machine_learning/rbf_network.zig").RBFNetwork;
pub const RBFConfig = @import("machine_learning/rbf_network.zig").Config;
pub const CenterMethod = @import("machine_learning/rbf_network.zig").CenterMethod;
pub const WidthMethod = @import("machine_learning/rbf_network.zig").WidthMethod;

// Sequence modeling algorithms
pub const HMM = @import("machine_learning/hidden_markov_model.zig").HMM;
pub const CRF = @import("machine_learning/crf.zig").CRF;

// Reinforcement learning algorithms
pub const QLearning = @import("machine_learning/q_learning.zig").QLearning;
pub const SARSA = @import("machine_learning/sarsa.zig").SARSA;
pub const ExpectedSARSA = @import("machine_learning/expected_sarsa.zig").ExpectedSARSA;
pub const ActorCritic = @import("machine_learning/actor_critic.zig").ActorCritic;
pub const REINFORCE = @import("machine_learning/reinforce.zig").REINFORCE;
pub const REINFORCEConfig = @import("machine_learning/reinforce.zig").Config;
pub const DQN = @import("machine_learning/dqn.zig").DQN;
pub const DDPG = @import("machine_learning/ddpg.zig").DDPG;
pub const PPO = @import("machine_learning/ppo.zig").PPO;
pub const TD3 = @import("machine_learning/td3.zig").TD3;
