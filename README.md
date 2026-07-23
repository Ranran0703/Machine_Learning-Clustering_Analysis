# Machine_Learning-Clustering_Analysis
# README.md

# Project Submission: Machine Learning & Clustering Analysis Pipeline
**Author:** Putri Maharani Fetra  
**Directory:** `Submission_Akhir_BMLP_Putri_Maharani_Fetra5/`  
**Status:** Completed & Verified  

---

##  Project Documentation & Criteria Mapping

This comprehensive documentation details the structural layout, methodology, and implementation pipelines embedded within this project repository, explicitly mapped according to the mandatory evaluation criteria.

---

##  Kriteria 1: Memuat Dataset dan Melakukan Exploratory Data Analysis (EDA)
The implementation pipeline initializes inside the main Jupyter Notebook (`[Clustering]_Submission_Akhir_BMLP_Putri_Maharani_Fetra5.ipynb`) by conducting a rigorous Exploratory Data Analysis (EDA) layer to evaluate data attributes and distributions:
* **`head()` Inspection:** Executed directly after loading the raw data matrix to review the first 5 records, verifying correct structural schema alignments, column boundaries, and basic structural sanity.
* **`info()` Architectural Review:** Invoked to output the exact schema overview, detailing the data type signatures (`int64`, `float64`, `object`), memory footprints, and confirming the total structural size of non-null fields per column vector.
* **`describe()` Summary Statistics:** Executed to extract full continuous descriptive characteristics, documenting statistical indices across the dataset including:
  * **Count:** Row totals verifying presence per feature.
  * **Mean & Median (50%):** Outlining data skewness and central tendencies.
  * **Standard Deviation (std):** Capturing underlying dispersion and variability metrics.
  * **Min & Max:** Pinpointing absolute boundaries.
  * **Quartiles (25%, 75%):** Mapping statistical distribution spreads.

---

##  Kriteria 2: Pembersihan dan Pra Pemrosesan Data (Data Preprocessing)
To prepare the dataset for algorithmic clustering, a robust data cleaning pipeline has been applied:
* **Missing Value & Duplicate Checks:** Used `isnull().sum()` to systematically inventory null counts per feature vector, combined with `duplicated().sum()` to calculate absolute redundant row duplications across the data frame.
* **Null Element Mitigation:** Handled incomplete structures by executing `dropna()` to drop invalid entries safely without distorting the global population variance.
* **Redundancy Exclusion:** Stripped out identical repeated vectors using `drop_duplicates()` to protect distance matrices from artificial dense weight inflation.
* **Metadata Dimension Stripping:** Dropped identifier metrics and temporal coordinates such as `TransactionID`, `AccountID`, `DeviceID`, `IPAddress`, `MerchantID`, and `TransactionDate`. Eliminating these high-cardinality nominal indexes prevents cluster drift caused by arbitrary non-predictive attributes.
* **Categorical Vector Quantization:** Applied `LabelEncoder()` from `sklearn.preprocessing` to transform text/categorical columns into discrete numerical representations, creating compatible numerical frameworks for subsequent distance computations.

---

##  Kriteria 3: Membangun Model Clustering
The clustering section isolates behavioral matrices via unsupervised structural grouping:
* **Clean Dataset Isolation:** Operates directly on the normalized, cleansed matrices generated from the Kriteria 2 pipeline.
* **Elbow Method Visualizations:** Integrated `KElbowVisualizer()` to plot distortion metrics across sequential value selections of $K$. This mathematically flags the inflection point ("elbow"), establishing an optimal cluster count that maximizes intra-cluster cohesion and inter-cluster separation.
* **K-Means Algorithmic Grouping:** Deployed `sklearn.cluster.KMeans()` with fixed initialization variables (`n_init=10`, explicitly controlled `random_state`) to form distinct, non-overlapping groups.
* **Automated Serial Model Export:** The final cluster state is systematically saved using `joblib.dump()` into a binary structure named `model_clustering` to facilitate programmatic evaluation, deployment, and reviewer verification.

---

##  Kriteria 4: Interpretasi Hasil Clustering
Post-clustering analysis validates and characterizes the generated user sub-populations:
* **Descriptive Feature Aggregation:** Calculated full mathematical metrics (**Mean, Min, Max**) across every quantitative attribute grouped by the assigned clusters.
* **Cluster Characteristic Definition:** Translated mathematical aggregates into concrete descriptive profiles, delineating segments based on spending tendencies, age group distributions, and activity scores.
* **Dataset Export with `Target` Column:** The final training dataset alongside its processed properties has been exported into `data_clustering.csv` and `data_clustering_inverse.csv`. The output contains a dedicated **`Target`** column mapped directly to the predicted cluster assignments.

---

##  Kriteria 5: Membangun Model Klasifikasi
A supervised learning layer is appended to automatically predict and assign cluster labels to incoming user pipelines:
* **Matrix Splitting:** Deployed `train_test_split()` to separate features from the generated `Target` labels, establishing dedicated training subsets and out-of-sample evaluation test segments.
* **Decision Tree Architecture:** Evaluated and constructed a supervised `Decision Tree` classifier using `sklearn.tree.DecisionTreeClassifier` to form predictable node-splitting logic.
* **Automated Classifier Serialization:** Executed `joblib.dump()` to serialize the trained tree architecture directly into `decision_tree_model.h5`, guaranteeing instant automated assessment capabilities for reviewers.

---

##  Repository File Structure
```text
Submission_Akhir_BMLP_Putri_Maharani_Fetra5/
├── [Clustering]_Submission_Akhir_BMLP_Putri_Maharani_Fetra5.ipynb  # Main End-to-End Pipeline
├── README.md                                                        # This Project Documentation File
├── data_clustering.csv                                             # Cleaned Features with "Target" Column
├── data_clustering_inverse.csv                                     # Inverse-Scaled Profiles for Reference
├── model_clustering                                                # Serialized K-Means Model Object
├── PCA_model_clustering.h5                                         # Trained PCA Transformer Object
├── decision_tree_model.h5                                          # Serialized Supervised Decision Tree Model
├── explore_RandomForest_classification.h5                          # Baseline Supervised Random Forest Model
└── tuning_classification.h5                                        # Hyperparameter-Tuned Classifier Archive
```
