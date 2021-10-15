# Detecting Text in Natural Image with Connectionist Text Proposal Network
The codes are used for implementing CTPN for scene text detection, described in: 

    Z. Tian, W. Huang, T. He, P. He and Y. Qiao: Detecting Text in Natural Image with
    Connectionist Text Proposal Network, ECCV, 2016.

Online demo is available at: [textdet.com](http://textdet.com)

These demo codes (with our trained model) are for text-line detection (without 
side-refinement part).

# Required hardware
You need a GPU. If you use CUDNN, about 1.5GB free memory is required. If you don't use CUDNN, you will need about 5GB free memory, and the testing time will slightly increase. Therefore, we strongly recommend to use CUDNN.

It's also possible to run the program on CPU only, but it's extremely slow due to the non-optimal CPU implementation.
# Required softwares
Python2.7, cython and all what Caffe depends on.

# How to run this code

1. Clone this repository with `git clone https://github.com/tianzhi0549/CTPN.git`. It will checkout the codes of CTPN and Caffe we ship.

2. Install the caffe we ship with codes bellow.
    * Install caffe's dependencies. You can follow [this tutorial](http://caffe.berkeleyvision.org/installation.html). *Note: we need Python support. The CUDA version we need is 7.0.*
    * Enter the directory `caffe`.
    * Run `cp Makefile.config.example Makefile.config`.
    * Open Makefile.config and set `WITH_PYTHON_LAYER := 1`. If you want to use CUDNN, please also set `CUDNN := 1`. Uncomment the `CPU_ONLY :=1` if you want to compile it without GPU.

      *Note: To use CUDNN, you need to download CUDNN from NVIDIA's official website, and install it in advance. The CUDNN version we use is 3.0.*
    * Run `make -j && make pycaffe`.

3. After Caffe is set up, you need to download a trained model (about 78M) from [Google Drive](https://drive.google.com/file/d/0B7c5Ix-XO7hqQWtKQ0lxTko4ZGs/view?resourcekey=0-_t07b9voEdZaFn3sxt9pTA) or [our website](http://textdet.com/downloads/ctpn_trained_model.caffemodel), and then populate it into directory `models`. The model's name should be ` ctpn_trained_model.caffemodel`.

4. Now, be sure you are in the root directory of the codes. Run `make` to compile some cython files.

5. Run `python tools/demo.py` for a demo. Or `python tools/demo.py --no-gpu` to run it under CPU mode.

# How to use other Caffe
If you may want to use other Caffe instead of the one we ship for some reasons, you need to migrate the following layers into the Caffe.
* Reverse
* Transpose
* Lstm

# License
The codes are released under the MIT License.
