# GraphMammoNet
Mammograms image detection using GraphMammoNet (MammoGNN),  GAT, GCN, and GraphConv based architectures. The code has been performed on both classification tasks of 4-class for Breast density types and 8-class for BIRAD scores.

# Dataset link
```
Please contact us to access both the original and preprocessing (edge detection transformed) data.
```
# File Structure and Working procedure
```
1. First apply edge detection accroding to the class-number (4 classes for Breast density types and 8 classes for BIRAD scores): Edge detection/edge_detection_<class-number>class.py
2. Then prepare graph-datasets using edge-preparation: Edge preparation/edge_preparation_<class-number>class.py
3. Finally edge preperation produces five kinds of dataset for graph classification:
  path name: .../GraphTrain/dataset/<dataset_name>/raw/<dataset_name>_<data_file>.txt. 
  <data_file> can be:
    
    1. A--> adjancency matrix 
    2. graph_indicator--> graph-ids of all node 
    3. graph_labels--> labels for all graph 
    4. node_attributes--> attribute(s) for all node 
    5. node_labels--> labels for all node
4. After all the graph datasets are created properly, run main.py. The graph datasets are loaded through dataloader.py and the model is called through model.py
```

# Citation
We have published our work entitled as "Edge detection and graph neural networks to classify mammograms: A case study with a dataset from Vietnamese patients" under the "Applied Soft Computing Journal". If this repository helps you in your research in any way, please cite our paper:
```bibtex
@article{DUONG2022109974,
title = {Edge detection and graph neural networks to classify mammograms: A case study with a dataset from Vietnamese patients},
journal = {Applied Soft Computing},
pages = {109974},
year = {2022},
issn = {1568-4946},
doi = {https://doi.org/10.1016/j.asoc.2022.109974},
url = {https://www.sciencedirect.com/science/article/pii/S1568494622010237},
author = {Linh T. Duong and Cong Q. Chu and Phuong T. Nguyen and Son T. Nguyen and Binh Q. Tran},
abstract = {Mammograms are breast X-ray images and they are used by doctors, among other purposes, as an effective means of detecting breast cancer. Screening mammography is crucial since it allows doctors to understand better the situation and have suitable intervention. The classification of medical modalities is a prerequisite for development of computer-aided diagnosis tools in healthcare, and various techniques have been proposed to automatically classify from mammography images. Though there have been several tools developed, they have been mostly validated with data collected from Western women. Based on our initial investigations, breast anatomy in Vietnamese women differs from that of Western women, due to denser breast tissue. In this paper, we propose MammoGNN – a practical solution to the classification of mammograms using the synergy between image processing techniques and graph neural networks. First, a well-founded edge detection algorithm was applied to provide input for the recommendation engine. Afterward, we empirically experimented to select suitable graph neural networks to manage the training and prediction. A mammogram dataset was curated from 2,351 Vietnamese women to validate the conceived tool. By several testing instances, MammoGNN obtains a maximum accuracy of 100%, precision and recall of 1.0 on independent and shuffle test sets for both classification of BI-RADS scores and breast density types. The experimental results also demonstrate that our proposed approach obtains an optimal prediction performance on the considered datasets, outperforming different baselines. We anticipate that the proposed approach can be deployed as a non-invasive pre-screening tool to assist doctors in performing their diagnosis activities.}
}
```
### Latest DOI

[![DOI](https://doi.org/10.1016/j.asoc.2022.109974)](https://doi.org/10.1016/j.asoc.2022.109974)

