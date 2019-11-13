# onYourMarks
A benchmark to help the decision making process over the Nvidia's Jetson TX2 platform.
It's designed to serve as a general purpouse benchmark, being easily
customizable for adding and adapting new models to the algorithm.

## Getting Started

To start with, it is needed to install all the dependencies. The script
has the requirement of using it with python 3.5 or superior.

After installing the python interpreter, it could be as simple as:

```
pip install -r requirements.txt
```

After the installation of initial packages, execute:

```
cd src/
python benchmark.py
```

The algorithm is responsible by downloading all the needed datasets
and/or models already defined under the `src/networks.py` file. If
you want to add your own custom models, generate a tensorflow frozen
graph and host it under this file.

## Troubleshooting

**Tensorflow GPU:** To install a specific version of Tensorflow, if the
jetpack installation failed by some reason, execute:

```
sudo pip3 install --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v$JP_VERSION tensorflow-gpu==$TF_VERSION+nv$NV_VERSION
```
Where `JP_VERSION` is the JetPack version of the specific tensorflow
version needed.
