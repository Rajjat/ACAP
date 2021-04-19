**Adaptive Clustering approach for Accident Prediction (ACAP)**  
ACAP compares accident prediction on different spatial aggregations. 

The repository contains two folders: 
1. Data preprocessing pipeline  
   The data preprocessing pipeline mainly process accident data from "Unfallatlas" and OpenStreetMap. It also implements various clustering pipelines,e.g., Grid        Growing, DBSCAN, Kmeans, SOM, and HDBSCAN.  
   a) Run 1_Accident_Extract_City.ipynb for extracting accident dataset for the specific city.  
   b) Run 2_train_testSplitAndNonaccGeneration.py for creating train and test data as well creating non-accident data.  
   c) Run 3_clustering/GG.ipynb for Grid growing aggregation whereas 3_clustering/Clustering_Algo.ipynb for clustering-based baselines.  
   d) Run 4_Aggregated_Spatial_Features/accident_spatialattributes.ipynb for creating acccident features for each grid/cluster.   
   e) 5_create_trainTestData folder consists overall pipeline for creating train and test data(accident data+ sptaial features) for 1x1 and clustering based               approach.   
   
2. Model Implementations  
   This folder consists of our model(ACAP) as well implementation of different baselines.  
   i) To run ACAP model, execute  
      python ACAP.py  
   ii) To run baselines(LR,GBC):  
        python GBCAndLR.py
   
  
 **Tools:-**  
   * Python(3.7)
   * Keras(2.3.1)
   * tensorflow(2.2.0)
   * scikit-learn(0.24.1)  

All the experiments are performed on GPU.
 
