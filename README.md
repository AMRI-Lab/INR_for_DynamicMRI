# INR_for_DynamicMRI
Official code for "Spatiotemporal implicit neural representation for unsupervised dynamic MRI reconstruction"
 
## 1. Environmental Requirements  
### To run the reconstruction demo, the following dependencies are required:  
* Python 3.10.X  ***(Important)***
* PyTorch 2.0.0
* torchkbnufft 1.4.0
* [tiny-cuda-nn 1.7](https://github.com/NVlabs/tiny-cuda-nn)
* imageio 2.18.0
* torchvision, tensorboard, h5py, scikit-image, tqdm, numpy, scipy

## 2. Sample Data
Download the sample data from [https://drive.google.com/file/d/1DIdtHcHUDEqx-qL4930-pz9mxCI8OYMR/view?usp=sharing](https://drive.google.com/file/d/1DIdtHcHUDEqx-qL4930-pz9mxCI8OYMR/view?usp=sharing) and put it into the root directory

## 3. Run the Demos
### To run the basic reconstruction demo, please use the following code:  
```python
python3 main.py -g 0 -s 13 -r -m
```

### To ablate relative L2 loss, please use the following code:  
```python
python3 main.py -g 0 -s 13 -m
```

### To ablate the coarse-to-fine strategy, please use the following code:  
```python
python3 main.py -g 0 -s 13 -r
```

### To run the interpolation demo, please use the following code:  
```python
python3 main_spatial_interp.py -g 0 -s 34 -r -m
```
or
```python
python3 main_temporal_interp.py -g 0 -s 34 -r -m
```

The rest of the parameters can be easily changed by adding arguments to the parser. 
The detailed definitions of the arguments can be found by: 
```python
python3 main.py -h
```
