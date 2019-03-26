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


## Applications
* modeling social influence
  * Deepinf:Modeling influence locality in large social networks,KDD 2018
* recommendation
  * Graph convolutional matrix completion,arXiv 2017
  * Geometric matrix completion with recurrent multi-graph neural networks,ICML 2017
  * Graph convolutional neural networks for web-scale recommender systems,KDD 2018
* science
  * Molecular Fingerprints
  * Protein Interface Preidction
  * chemistry
  * physics Systems
  * disease or drug prediction 
  * Disease Classification
  * Side Effects Preidction
* natural language processing (NLP)
  * Text classification
  * Sequence Labeling (POS, NER)
  * Sentiment classification
  * Semantic role labeling
  * Neural machine translation
  * Relation extraction
  * Event extraction
  * AMR to text generation
  * Multi-hop reading comprehension
* image
  * Social Relationship Understanding
  * Image classification
  * Visual Question Answering
  * Region Recognition
  * Semantic Segmentation
  * computer vision
  * visual scene understanding tasks
* knowledge graph
  * Translating embeddings for modeling multi-relational data,NIPS 2013
  * Representation learning for visual-relational knowledge graphs,arXiv 2017
  * Knowledge transfer for out-of-knowledge-base entities : A graph neural network approach,IJCAI 2017
* traffic forecasting
  * High-order graph convolutional recurrent neural network: A deep learning framework for network-scale traffic learning and forecasting,arXiv 2018
  * Spatio-temporal graph convolutional networks:A deep learning framework for traffic forecasting,IJCAI 2018
  * Diffusion convolutional recurrent neural network: Data-driven traffic forecasting,ICLR 2018
* program induction
* few-shot learning
* multi-agent systems
* solving graph-based NP problems

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


## some related resources
* [Awesome-Graph-Neural-Networks](https://github.com/nnzhan/Awesome-Graph-Neural-Networks)
* [GNNPaper:Must-read papers](https://github.com/thunlp/GNNPapers) 清华大学NLP组
