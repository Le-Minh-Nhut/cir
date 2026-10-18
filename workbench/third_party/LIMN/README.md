# Self-Training Boosted Multi-Factor Matching Network for Composed Image Retrieval [IEEE TPAMI 2024]

## Authors

**Haokun Wen**<sup>1</sup>, **Xuemeng Song**<sup>2</sup>\*, **Jianhua Yin**<sup>2</sup>, **Jianlong Wu**<sup>1</sup>, **Weili Guan**<sup>1</sup>, **Liqiang Nie**<sup>1</sup>\*

<sup>1</sup> Harbin Institute of Technology (Shenzhen), Shenzhen, China  
<sup>2</sup> Shandong University, Qingdao, China  
\* Corresponding authors

## Links

- **Paper**: [IEEE TPAMI](https://doi.org/10.1109/TPAMI.2023.3346434)
- **Checkpoints**: [HaokunWen/LIMN](https://huggingface.co/HaokunWen/LIMN/tree/main)

---

## Running Environments

- GPU: 1 × NVIDIA A100 40G
- CUDA 12.4
- torch == 1.12.1
- torchvision == 0.13.1
- open-clip-torch == 2.20.0

---

## Repository Structure

```text
.
├── LIMN/                  # Source code for LIMN
├── LIMN+/                 # Source code for LIMN+
│   └── pseudo_data/       # Pre-generated pseudo triplets for self-training
├── correct_files/         # Spell-corrected modification texts
│   ├── correction_dict_dress.json
│   ├── correction_dict_shirt.json
│   ├── correction_dict_toptee.json
│   └── correction_dict_shoes.json
├── requirements.txt
└── LICENSE
```

---

## Dataset Preparation

### FashionIQ

1. Download the dataset from the [official repository](https://github.com/XiaoxiaoGuo/fashion-iq) or from [Google Drive](https://drive.google.com/drive/folders/14JG_w0V58iex62bVUHSBDYGBUECbDdx9?usp=sharing) (provided by CosMo authors). Store it at `[your path]/fashion_iq_data`.
2. Resize images using [`resize_images.py`](https://github.com/XiaoxiaoGuo/fashion-iq/blob/master/start_kit/resize_images.py). Save resized images to `[your path]/fashion_iq_data/resized_image`.
3. Spell-corrected modification texts are provided at `correct_files/correction_dict_{dress/shirt/toptee}.json`.

### Shoes

1. The original Shoes dataset images are no longer available from the official source. Download from [Google Drive](https://drive.google.com/file/d/18DEWXvuyp2vXHv4tAw6fcD2ehEtrvyIL/view?usp=sharing). Unzip and place images by name inside `~/shoes_data/womens_*/` folders.
2. Spell-corrected modification texts are provided at `correct_files/correction_dict_shoes.json`.

### CIRR

Download the CIRR dataset following the instructions in the [official repository](https://github.com/Cuberick-Orion/CIRR).

> **Note**: CIRR test results must be submitted through the [official evaluation website](https://cirr.cecs.anu.edu.au/). Use `cirr_test_submission.py` to generate submission files.

### Fashion200K

Download the Fashion200K dataset following the instructions in the [official repository](https://github.com/xthan/fashion-200k).

---

## Usage

### LIMN

Source code is located in the `./LIMN` folder. Checkpoints are saved as `checkpoint/0_{dataset}_best_model.pt`.

#### FashionIQ

```bash
# Run separately for each category: dress / shirt / toptee
python train.py --dataset dress --prefix_num 8 --norm 1 --peer_ranking 1 --model_dir './fashioniq'
python train.py --dataset shirt --prefix_num 8 --norm 1 --peer_ranking 1 --model_dir './fashioniq'
python train.py --dataset toptee --prefix_num 8 --norm 1 --peer_ranking 1 --model_dir './fashioniq'
```

#### Shoes

```bash
python train.py --dataset shoes --prefix_num 8 --norm 1 --peer_ranking 1 --model_dir './shoes'
```

#### CIRR

```bash
python train.py --dataset cirr --prefix_num 10 --norm 1 --peer_ranking 1 --model_dir './cirr'
python cirr_test_submission.py
```

#### Fashion200K

```bash
python train.py --dataset fashion200k --prefix_num 8 --norm 1 --peer_ranking 1 --model_dir './fashion200k'
```

---

### LIMN+

Source code is located in the `./LIMN+` folder. Pre-generated pseudo triplets are provided in `LIMN+/pseudo_data/`. The number in `augment_data_x_1/2.json` indicates the self-training iteration number.

#### FashionIQ — Dress

```bash
python train.py --dataset 'dress' --prefix_num 8 --norm 1 --peer_ranking 1 \
  --interval 3 --p_origin 0.3 --p_aug 0.6 --augment_iteration 2 --model_dir './dress'
```

#### FashionIQ — Shirt

```bash
python train.py --dataset 'shirt' --prefix_num 8 --norm 1 --peer_ranking 1 \
  --interval 5 --p_origin 0.3 --p_aug 0.6 --augment_iteration 2 --model_dir './shirt'
```

#### FashionIQ — Top & Tee

```bash
python train.py --dataset 'toptee' --prefix_num 8 --norm 1 --peer_ranking 1 \
  --interval 4 --p_origin 0.3 --p_aug 0.6 --augment_iteration 2 --model_dir './toptee'
```

#### Shoes

```bash
python train.py --dataset 'shoes' --prefix_num 8 --norm 1 --peer_ranking 1 \
  --interval 5 --p_origin 0.2 --p_aug 0.5 --augment_iteration 1 --model_dir './shoes'
```

#### CIRR

```bash
python train.py --dataset 'cirr' --prefix_num 10 --norm 1 --peer_ranking 1 \
  --interval 5 --p_origin 0 --p_aug 0.3 --augment_iteration 1 --model_dir './cirr'
python cirr_test_submission.py
```

---

## Checkpoints

Pre-trained model weights are available on Hugging Face: [HaokunWen/LIMN](https://huggingface.co/HaokunWen/LIMN/tree/main).

Download and place checkpoints under the `checkpoint/` directory within the corresponding `LIMN/` or `LIMN+/` folder.

---

## Citation

If you find this work useful, please cite:

```bibtex
@article{LIMN,
  author  = {Haokun Wen and Xuemeng Song and Jianhua Yin and Jianlong Wu and Weili Guan and Liqiang Nie},
  title   = {Self-Training Boosted Multi-Factor Matching Network for Composed Image Retrieval},
  journal = {{IEEE} Transactions on Pattern Analysis and Machine Intelligence},
  volume  = {46},
  number  = {5},
  pages   = {3665--3678},
  year    = {2024}
}
```
