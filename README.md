# Graph Neural Network for Node Classification


 <div align="center">
    <a href="https://colab.research.google.com/github/reshalfahsi/node-classification/blob/master/Node_Classification.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="colab"></a>
    <br />
 </div>


A graph neural network (GNN) is a type of neural network leveraged to handle graph data. One kind of graph data is a single graph that is large enough to contain a myriad of nodes. Later, we can attribute each node to well-qualified features and discriminate them accordingly. Then, by means of GNN, we can perform node classification on this large graph. The CORA dataset, the publicly available dataset for node classification on a large graph, is used in this tutorial. The graph feature extractor utilized in this tutorial consists of a sequence of ``ResGatedGraphConv``, ``SAGEConv``, and ``TransformerConv``, which are implemented by [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/index.html). The final classifier comprises MLP.


## Experiment


To run the code for this project, please click [here](https://github.com/reshalfahsi/node-classification/blob/master/Node_Classification.ipynb).


## Result

## Quantitative Result

The table below exhibit the quantitative performance of the model on the test data.

Metrics | Score |
------------ | ------------- |
Accuracy | 0.742 |
Loss | 2.695 |



## Accuracy and Loss Curve

<p align="center"> <img src="https://github.com/reshalfahsi/node-classification/blob/master/assets/accuracy_curve.png" alt="accuracy_curve" > <br /> GNN's training and validation accuracy curve. </p>

<p align="center"> <img src="https://github.com/reshalfahsi/node-classification/blob/master/assets/loss_curve.png" alt="loss_curve" > <br /> GNN's training and validation loss curve. </p>


## Qualitative Result

The qualitative result is provided as follows:

<p align="center"> <img src="https://github.com/reshalfahsi/node-classification/blob/master/assets/qualitative_result.gif" alt="qualitative_result" > <br /> The visualization of the embedding space of the nodes in the large graph in the course of training process. </p>


## Credit

- [Residual Gated Graph ConvNets](https://arxiv.org/pdf/1711.07553.pdf)
- [Inductive Representation Learning on Large Graphs](https://arxiv.org/pdf/1706.02216.pdf)
- [Masked Label Prediction: Unified Message Passing Model for Semi-Supervised Classification](https://arxiv.org/pdf/2009.03509.pdf)
- [CORA dataset](https://relational.fit.cvut.cz/dataset/CORA)
- [Node Classification on Large Knowledge Graphs](https://colab.research.google.com/drive/1LJir3T6M6Omc2Vn2GV2cDW_GV2YfI53_)
- [Graph attention network (GAT) for node classification](https://keras.io/examples/graph/gat_node_classification/)
- [PyTorch Geometric](https://github.com/pyg-team/pytorch_geometric)
- [PyTorch Lightning](https://lightning.ai/docs/pytorch/latest/)
