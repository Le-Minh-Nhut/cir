# Target-Guided Composed Image Retrieval [ACM MM 2023]

## Authors
 
**Haokun Wen**<sup>1</sup>, **Xian Zhang**<sup>1</sup>, **Xuemeng Song**<sup>2</sup>\*, **Yinwei Wei**<sup>3</sup>, **Liqiang Nie**<sup>1</sup>\*
 
<sup>1</sup> Harbin Institute of Technology (Shenzhen), Shenzhen, China  
<sup>2</sup> Shandong University, Qingdao, China  
<sup>3</sup> Monash University, Melbourne, Australia  
\* Corresponding authors

## Links
 
- **Paper**: [ACM DL](https://dl.acm.org/doi/10.1145/3581783.3611817)
- **Checkpoints**: [Download Link](https://drive.google.com/file/d/1OdZTtJqy-RTpYXBaq5ThmH3IBnGvYCyi/view)

## Key dependencies
```
clip==0.2.0
matplotlib==3.5.1
numpy==1.22.3
pandas==1.4.2
Pillow==9.0.1 / 10.0.0
seaborn==0.12.2
torch==1.7.0
torchvision==0.8.0
tqdm==4.65.0
```

## Dataset Preparation
 
### FashionIQ & Shoes
 
Spell-corrected data files for FashionIQ and Shoes are provided in correction_dict folder.
 
### CIRR
 
For CIRR, test results must be submitted through the official [CIRR evaluation website](https://cirr.cecs.anu.edu.au/). Submission files can be generated using:
 
```bash
python cirr_test_submission.py
```
 
---
 
## Usage
 
### Training
 
#### FashionIQ
 
```bash
python train.py --dataset 'fashioniq' --model_dir <output_dir> \
  --mu_ 0.1 --nu_ 10 --kappa_ 0.5 --tau_ 0.1 --P 4 --Q 8
```
 
#### Shoes
 
```bash
python train.py --dataset 'shoes' --model_dir <output_dir> \
  --mu_ 0.05 --nu_ 5 --kappa_ 0.5 --tau_ 0.1 --P 3 --Q 6
```
 
#### CIRR
 
```bash
python train.py --dataset 'cirr' --model_dir <output_dir> \
  --mu_ 0.1 --nu_ 1 --kappa_ 0.1 --tau_ 0.05 --P 4 --Q 8
```

 
---
 
## Citation
 
If you find this work useful, please cite:
 
```bibtex
@inproceedings{wen2023target,
  title={Target-Guided Composed Image Retrieval},
  author={Wen, Haokun and Zhang, Xian and Song, Xuemeng and Wei, Yinwei and Nie, Liqiang},
  booktitle={Proceedings of the ACM International Conference on Multimedia},
  pages={915--923},
  year={2023}
}
```

---
 
## License
 
This program is licensed under the GNU General Public License 3.0.  
See [https://www.gnu.org/licenses/gpl-3.0.html](https://www.gnu.org/licenses/gpl-3.0.html) for details.  
Any derivative work must also be licensed under the GNU GPL as published by the Free Software Foundation, either Version 3 or (at your option) any later version.
