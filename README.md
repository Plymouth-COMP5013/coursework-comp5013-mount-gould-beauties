# COMP5013 PyTorch STGCN

A custom STGCN repository for COMP5013, built using PyTorch and, more specifically, PyTorch Geometric Temporal.

## Original Paper

The STGCN model we're building from was conceived by this research team in 2018.

> Bing Yu*, Haoteng Yin*, Zhanxing Zhu. [Spatio-temporal Graph Convolutional Networks: A Deep Learning Framework for Traffic Forecasting](https://www.ijcai.org/proceedings/2018/0505). In _Proceedings of the 27th International Joint Conference on Artificial Intelligence (IJCAI)_, 2018

## Dataset

**[PeMSD7](http://pems.dot.ca.gov/)** is the dataset used by the original research team, and the one used for this piece of work. It was collected from Caltrans Performance Measurement System (PeMS) in real-time by over 39,000 sensor stations, deployed across the major metropolitan areas of California state highway system. The dataset is aggregated into 5-minute intervals from 30-second data samples.

The team randomly selected a medium and a large scale among the District 7 of California containing 228 and 1,026 stations, labelled as PeMSD7(M) and PeMSD7(L), respectively, as data sources. The time range of PeMSD7 dataset is in the weekdays of May and June of 2012. Overall, there are 44 days of historical traffic speed data captured.

`PeMSD7_W_288.csv` is the adjacency matrix of the 288 sensors in the PeMSD7 dataset. The adjacency matrix is a 288x288 matrix, where each element represents the distance between two sensors. This matrix makes up the spatial element of the STGCN model, and can serve as a weighting matrix for the graph convolutional layers.

`PeMSD7_V_288.csv` is the traffic speed data of the 288 sensors in the PeMSD7 dataset. With 44 days of data, the dataset contains 44 x 24 x 12 = 12,672 samples. Each sample is a 288-dimensional vector, where each element represents the traffic speed of a sensor at a given time. This matrix makes up the temporal element of the STGCN model, and can serve as a feature matrix for the graph convolutional layers. Note that each row is a new record and each column is a sensor in the network.

For version control, the data files are compressed into a zip file. You will need to unzip those files with a tool of your choice (e.g. `unzip` on Linux or `7-Zip` on Windows) before running the code. Keep all data files are located in the `dataset` directory.

### Test Data Import

In order to test that the data has been unzipped and is able to be used correctly, run `test.py`. You should see the following output in your terminal:

```bash
Training set size: 10128
Testing set size: 1266
Validation set size: 1266
<class 'torch_geometric.data.data.Data'>
Data(x=[228, 12], edge_index=[2, 51722], edge_attr=[51722], y=[228, 1])
```
