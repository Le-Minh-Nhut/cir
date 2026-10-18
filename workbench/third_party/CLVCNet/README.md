# `Comprehensive Linguistic-Visual Composition Network for Image Retrieval [SIGIR 2021]`

## Authors

**Haokun Wen**<sup>1</sup>, **Xuemeng Song**<sup>1</sup>\*, **Xin Yang**<sup>1</sup>, **Yibing Zhan**<sup>2</sup>, **Liqiang Nie**<sup>1</sup>\*

<sup>1</sup> `Shandong University`  
<sup>2</sup> `JD Explore Academy`  
\* Corresponding author

## Links
 
- **Paper**: [ACM DL](https://dl.acm.org/doi/abs/10.1145/3404835.3462967)

---

## 1. Running Environments
- Python 3.7.6
- PyTorch 1.6.0
- GPU: TITAN XP
- OS: Ubuntu 14.04.6 LTS

## 2. Usage

#### FashionIQ
 
Create directories `dress/`, `shirt/`, and `toptee/` to save the outputs.
Note that for the FashionIQ dataset, we first resize the downloaded images follow the [XiaoxiaoGuo](https://github.com/XiaoxiaoGuo/fashion-iq/tree/master/start_kit) by resize_images.py.
 
```bash
# Dress
python train.py --dataset fashioniq --name dress --seed 599 \
  --max_decay_epoch 20 --img_weight 1.0 --class_weight 1.0 \
  --mul_kl_weight 1.0 --model_dir ./dress --num_epochs 50
 
# Shirt
python train.py --dataset fashioniq --name shirt --seed 599 \
  --max_decay_epoch 20 --img_weight 1.0 --class_weight 1.0 \
  --mul_kl_weight 1.0 --model_dir ./shirt --num_epochs 50
 
# Top & Tee
python train.py --dataset fashioniq --name toptee --seed 599 \
  --max_decay_epoch 20 --img_weight 1.0 --class_weight 1.0 \
  --mul_kl_weight 1.0 --model_dir ./toptee --num_epochs 50
```
 
#### Shoes
 
Create directory `shoes/` to save the outputs.
 
```bash
python train.py --dataset shoes --seed 6195 \
  --max_decay_epoch 30 --img_weight 1.0 --class_weight 1.0 \
  --mul_kl_weight 1.0 --model_dir ./shoes --num_epochs 50
```
 
#### Fashion200K
 
Create directory `fashion200k/` to save the outputs.
 
```bash
python train.py --dataset fashion200k --seed 6195 --num 1 \
  --img_weight 1.0 --class_weight 1.0 \
  --mul_kl_weight 1.0 --model_dir ./fashion200k --num_epochs 40
```
 
### Testing
 
```bash
python test.py
```
 
---

## 3. Checkpoints
* **Pre-trained models**: [Google Drive Link](https://drive.google.com/file/d/159rBhWyhkLN7sXAi8iyW_ljzFNLJinKa/view?usp=sharing)

## Citation
 
If you find this work useful, please cite:
 
```bibtex
@inproceedings{wen2021comprehensive,
  title={Comprehensive Linguistic-Visual Composition Network for Image Retrieval},
  author={Wen, Haokun and Song, Xuemeng and Yang, Xin and Zhan, Yibing and Nie, Liqiang},
  booktitle={Proceedings of the International ACM SIGIR Conference on Research and Development in Information Retrieval},
  pages={1369--1378},
  year={2021}
}
```
 
---
 
## License
 
This project is released under the Apache License 2.0.
