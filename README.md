# Graph-Neural-Networks
All materials related to GNN

## Related survey papers
* Deep Learning on Graphs: A Survey, arXiv 2018
* A Comprehensive Survey on Graph Neural Networks,arXiv 2018
* Graph Neural Networks: A Review of Methods and Applications,arXiv 2018
* Relational inductive biases, deep learning, and graph networks,arXiv 2018


## Motivation of GNN
* The first motivation of GNNs roots in convolutional neural networks (CNNs)
* The other motivation comes from graph embedding, which learns to represent graph nodes, edges or subgraphs in low-dimensional vectors.

## GNN worth investigating
* GNNs propagate on each node respectively, ignoring the input order of nodes
* GNNs can do propagation guided by the graph structure instead of using it as part of features
* GNNs explore to generate the graph from non structural data like scene pictures and story documents, which can be a powerful neural model for further high-level AI.

## Challenges of traditional deep learning on graphs
* Irregular domain
* Varying structures and tasks
* Scalability and parallelization
* Interdiscipline

## General Frameworks
* Message Passing Neural Networks(MPNN)
* Non-local Neural Networks(NLNN)
* Graph Networks(GN)

## Taxonomy of Deep Learning methods on graphs
* Graph Neural Networks
* Graph Convolutional Networks
  * Spectral-based
  * Spatial-based
  * Pooling modules
* Graph Auto-encoders
  * Auto-encoders
  * Variational Auto-encoders
* Graph Attention Networks
* Graph Generative Networks
* Graph Spatial-Temporal Networks
* Graph Recurrent Neural Networks
* Graph Reinforcement Learning

## Datasets
* Citation Networks
  * Cora (Collective classification in network data,AI magazine,2008)
  * Citeseer (Collective classification in network data,AI magazine,2008)
  * Pubmed (Collective classification in network data,AI magazine,2008)
  * [DBLP](aminer.org/citation)
* Social Networks
  * BlogCatalog (Relational learning via latent social dimensions,KDD 2009)
  * Reddit (representation learning on large graphs,NIPS 2017)
  * [Epinions](www.epinions.com)
* Chemical/Biological Graphs
  * PPI (Predicting multicellular function through multi-layer tissue networks,Bioinformatics 2017)
  * NCI-1 (Comparison of descriptor spaces for chemical compound retrieval and classification,KIS 2008)
  * NCI-109 (Comparison of descriptor spaces for chemical compound retrieval and classification,KIS 2008)
  * MUTAG (Structure-activity relationship of mutagenic aromatic and heteroaromatic nitro compounds. correlation with molecular orbital energies and hydrophobicity,Journal of medicinal chemistry,1991)
  * D&D (Distinguishing enzyme structures from non-enzymes without alignments,Journal of molecular biology 2003)
  * QM9 (Quantum chemistry structures and properties of 134 kilo molecules,Scientific data 2014)
  * [tox21](tripod.nih.gov/tox21/challenge/)
* Unstructured Graphs
  * [MNIST](yann.lecun.com/exdb/mnist/)
  * [Wikipedia](www.mattmahoney.net/dc/textdata)
  * 20NEWS (A probabilistic analysis of the rocchio algorithm with tfidf for text categorization.Carnegie-mellon univ pittsburgh pa dept of computer science, Tech. Rep., 1996)
* Others
  * METR-LA (Big data and its technical challenges,Communications of the ACM 2014)
  * [Movie-Lens1M](grouplens.org/datasets/movielens/1m/)
  * Nell (Toward an architecture for never-ending language learning,AAAI 2010)

## Open-source Implementations
* [ChebNet](https://github.com/mdeff/cnn_graph)
* [1stChebNet](https://github.com/tkipf/gcn)
* [GGNNs](https://github.com/yujiali/ggnn)
* [SSE](https://github.com/Hanjun-Dai/steady_state_embedding)
* [GraphSage](https://github.com/williamleif/GraphSAGE)
* [LGCN](https://github.com/williamleif/GraphSAGE)
* [SplineCNN](https://github.com/rusty1s/pytorch_geometric)
* [GAT](https://github.com/PetarV-/GAT)
* [GAE](https://github.com/limaosen0/Variational-Graph-Auto-Encoders)
* [ARGA](https://github.com/Ruiqi-Hu/ARGA)
* [DNGR](https://github.com/ShelsonCao/DNGR)
* [SDNE](https://github.com/suanrong/SDNE)
* [DRNE](https://github.com/tadpole/DRNE)
* [GraphRNN](https://github.com/snap-stanford/GraphRNN)
* [DCRNN](https://github.com/liyaguang/DCRNN)
* [CNN-GCN](https://github.com/VeritasYin/STGCN_IJCAI-18)
* [ST-GCN](https://github.com/yysijie/st-gcn)
* [Structural RNN](https://github.com/asheshjain399/RNNexp)
* [Bilinear GNN](https://github.com/zhuhm1996/bgnn)
* [PyGAS: Auto-Scaling GNNs in PyG](https://github.com/rusty1s/pyg_autoscale)


## Platform
* [AliGraph: A comprehensive graph neural network platform](https://github.com/alibaba/graph-learn)
* [DistDGL: Distributed graph neural network training for billion-scale graphs](https://github.com/dmlc/dgl/tree/master/python/dgl/distributed)
* [DGL is an easy-to-use, high performance and scalable Python package for deep learning on graphs](https://github.com/dmlc/dgl)

## Applications
* traffic forecasting
  * [TrafficStream: A Streaming Traffic Flow Forecasting FrameworkBased on Graph Neural Networks and Continual Learning（IJCAI 2021)](https://github.com/AprLie/TrafficStream)
  * [Spatial-Temporal Graph ODE Neural Network(2021 KDD)](https://github.com/square-coder/STGODE)
* recommendation
  * [Graph4Rec: A Universal and Large-scale Toolkit with Graph Neural Networks for Recommender Systems](https://github.com/PaddlePaddle/PGL/tree/graph4rec/apps/Graph4Rec)
* differential privacy
  * [differential privacy](https://github.com/tensorflow/privacy)

## Future directions
* Different types of graphs
* Dynamic graphs
* Interpretability
* Compositionality
* Go Deep
* Receptive Field
* Scalability
* Shallow Structure(graph neural net works are always shallow, most of which are no more than three layers.)
* Non-Structural Scenarios
* Green deep learning
* Low resource learning(FSL and ZSL)

## GNN application for specific field
 * [NLP with GNN](https://github.com/icoxfog417/graph-convolution-nlp)

## some related resources
* [Awesome Graph Neural Networks](https://github.com/nnzhan/Awesome-Graph-Neural-Networks)
* [GNNPaper:Must-read papers](https://github.com/thunlp/GNNPapers) 清华大学NLP组
* [GNN相关的一些论文以及最新进展](https://github.com/jdlc105/Must-read-papers-and-continuous-tracking-on-Graph-Neural-Network-GNN-progress)
* [Literature of Deep Learning for Graphs](https://github.com/DeepGraphLearning/LiteratureDL4Graph)
* [Graph-based deep learning literature](https://github.com/naganandy/graph-based-deep-learning-literature)
* [spatio temporal-paper-list(graph convolutional)](https://github.com/Eilene/spatio-temporal-paper-list)
* [Python package built to ease deep learning on graph, on top of existing DL frameworks](https://github.com/dmlc/dgl)
* [对于GNN综述文章的一个整理](https://github.com/ShiYaya/graph)
* [Geometric Deep Learning Extension Library for PyTorch](https://github.com/rusty1s/pytorch_geometric)
* [关于GNN的pytorch模型示例](https://github.com/LYuhang/GNN_Review)
* [Graph Neural Networks for Natural Language Processing](https://github.com/svjan5/GNNs-for-NLP)
* [Graph Neural Network for Traffic Forecasting](https://github.com/jwwthu/GNN4Traffic)
* [self-supervised learning on Graph Neural Networks](https://github.com/ChandlerBang/awesome-self-supervised-gnn)
* [awesome auto graph learning](https://github.com/THUMNLab/awesome-auto-graph-learning)
* [A Survey of Pretraining on Graphs: Taxonomy, Methods, and Applications](https://github.com/junxia97/awesome-pretrain-on-graphs)
* [Reinforcement learning on graphs: A survey](https://github.com/neunms/Reinforcement-learning-on-graphs-A-survey)
* [A Python Library for Graph Outlier Detection (Anomaly Detection)](https://github.com/pygod-team/pygod/)


## Researchers and Groups
* Data Mining
  * [Shirui Pan](https://shiruipan.github.io/)
  * [Hongzhi Yin](https://sites.google.com/view/hongzhi-yin/home)
  * [Bin Cui](http://net.pku.edu.cn/~cuibin/)
* CV
  * [Hengtao Shen](https://cfm.uestc.edu.cn/~shenht/)
