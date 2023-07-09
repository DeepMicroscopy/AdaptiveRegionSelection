# AdaptiveRegionSelection

This repository provides an Active Learning (AL) framework for selecting potentially informative annotation regions on 
Whole Slide Images (WSIs), with the goal of achieving full annotation performance with a minimum of annotated area.

For details, please see "Adaptive Region Selection for Active Learning in Whole Slide Image Semantic Segmentation".

## Active Learning (AL)
You may set "sampling_strategy" in [user_define.py](code/user_define.py):
* "full" (full annotation benchmark)
* "random"
* "uncertainty_standard"
* "uncertainty_non_square"
* "uncertainty_adapt"

You may set "n_query" and "region_size" in [user_define.py](code/user_define.py) to define AL parameters,
and "CYCLES" in [experiments.py](code/experiments.py) to define the number of conducted AL cycles.
```python
python experiments.py
```
## Conda Environment
We use Fastai_v1 for implementation. We observe some package conflicts while building the conda env, 
you may install as following:
```commandline
conda create -n fastai_v1_38 python=3.8
source activate fastai_v1_38
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
conda install -c fastai fastai=1.0.61
conda install pip
pip install -r requirements.txt
conda install pixman=0.40.0
conda install -c conda-forge openslide
```

## Full Annotation Benchmark
We trained on the fully-annotated data to validate our segmentation framework. In [full annotation benchmark](full%20annotation%20benchmark),
you may find two trained models.
