# CSMCIR: CoT-Enhanced Symmetric Alignment with Memory Bank for Composed Image Retrieval

:tada::tada::tada:
This is the implementation of CSMCIR proposed by our paper《CSMCIR: CoT-Enhanced Symmetric Alignment with Memory Bank for Composed Image Retrieval》.**(ACL206 Findings)**
### Prerequisites

	
The following commands will create a local Anaconda environment with the necessary packages installed.

```bash
conda create -n csmcir -y python=3.9
conda activate csmcir
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2
pip install -r requirements.txt
```


### Data Preparation

To properly work with the codebase FashionIQ and CIRR datasets should have the following structure:

```
project_base_path
└───  fashionIQ_dataset
      └─── captions
            | cap.dress.test.json
            | cap.dress.train.json
            | cap.dress.val.json
            | ...
            
      └───  images
            | B00006M009.jpg
            | B00006M00B.jpg
            | B00006M6IH.jpg
            | ...
            
      └─── image_splits
            | split.dress.test.json
            | split.dress.train.json
            | split.dress.val.json
            | ...

└───  cirr_dataset  
       └─── train
            └─── 0
                | train-10108-0-img0.png
                | train-10108-0-img1.png
                | train-10108-1-img0.png
                | ...
                
            └─── 1
                | train-10056-0-img0.png
                | train-10056-0-img1.png
                | train-10056-1-img0.png
                | ...
                
            ...
            
       └─── dev
            | dev-0-0-img0.png
            | dev-0-0-img1.png
            | dev-0-1-img0.png
            | ...
       
       └─── test1
            | test1-0-0-img0.png
            | test1-0-0-img1.png
            | test1-0-1-img0.png 
            | ...
       
       └─── cirr
            └─── captions
                | cap.rc2.test1.json
                | cap.rc2.train.json
                | cap.rc2.val.json
                
            └─── image_splits
                | split.rc2.test1.json
                | split.rc2.train.json
                | split.rc2.val.json
```
#### Shoes dataset:
For Shoes dataset, you can refer to DQU-CIR(https://github.com/iLearn-Lab/SIGIR24-DQU-CIR)

#### LaSCo dataset:
For LaSCo dataset, you can refer to LaSCo(https://github.com/levymsn/LaSCo)

#### Qwen captions:
The qwen-generated captions of CIRR, FashionIQ and Shoes can be available in https://huggingface.co/peng12138/CSMCIR, put them under the corresponding data directory

#### Note: Our further experiments find that our model is stable for captions, and even without them can also achieve good performance.

### Training


```sh
python src/blip_fine_tune_2_csmcir.py \
   --dataset {'CIRR' or 'FashionIQ' or 'Shoes' or 'Lasco'} \
   --blip-model-name 'blip2_cir_align_prompt_csmcir' \
   --num-epochs {'10' for CIRR, fashionIQ, Shoes, '5' for Lasco} \
   --num-workers 4 \
   --learning-rate {'1e-5'} \
   --batch-size 128 \
   --transform targetpad \
   --target-ratio 1.25  \
   --save-training \
   --save-best \
   --validation-frequency 1 
```

### Evaluation


```sh
python src/blip_validate_csmcir.py \
   --dataset {'CIRR' or 'FashionIQ' or 'Shoes' or 'Lasco'} \
   --blip-model-name {trained model name} \
   --model-path {for path} 
```

### CIRR Testing


```sh
python src/cirr_test_submission_csmcir.py \
   --blip-model-name {trained model name} \
   --model-path {for path} \
```

### Checkpoints
Our model checkpoints are available at https://huggingface.co/peng12138/CSMCIR.

### Acknowledgement
Our implementation is based on SPRC(https://github.com/chunmeifeng/SPRC).

