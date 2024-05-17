# SCULPT_release (CVPR 2024)
The official codebase for SCULPT: Shape-Conditioned Unpaired Learning of Pose-dependent Clothed and Textured Human Meshes

**[Project Website](https://sculpt.is.tue.mpg.de)** | **[Dataset Download](https://sculpt.is.tue.mpg.de/download.php)** | **[Arxiv Paper](https://arxiv.org/pdf/2308.10638v2)** | **[Video](https://youtu.be/KbVp30eLtT8)**

## Training


First clone the github repo.

```
git clone https://github.com/soubhiksanyal/SCULPT_release.git
cd SCULPT_release
```
Install the packages and the corresponding versions as mentioned in the requirements.txt file.

```
python3 -m venv SCULPT
source SCULPT/bin/activate
pip install -r requirements.txt
```

Install the following version of PyTorch. The training and inference code are tested on V100 and A100 GPUs. We have trained our models with 8 GPUs for five/six days for getting the reported result.

```
torch                    1.13.1
torchaudio               0.13.1
torchmetrics             0.11.1
torchvision              0.14.1
```
Create a data folder inside the main directory. 

```
mkdir data
```

Download and extract all the data from the project website and place them into the data folder.

**Do not unzip** `RGB_with_same_pose_white_16362_withclothinglabel_withnormals_withcolors_MODNetSegment_withalpha.zip` which is contains all the preprocessed images and annotations to train SCULPT.  


Then run the following command to start training

```
sh trainer_cluster_mul.sh
```

To train SCULPT with a new dataset, follow the script provided by dataset_tool.py. But one first needs to compute the clothing type and clothing color for the new data as describe in the main paper. We will add the scripts for these feature computations in a future update.

We already provide the checkpoint for the trained geometry generator which requires addtional five days to train. 

We also provide the raw fashion images (512x512) and their annotations in case one wants to train their own model for academic research.

## Inference 

First, create a data folder. Next, download and extract all the data from the project website and place them into the data folder. Then, run the following command to generate the meshes and renderings used in the main paper and the video. 

```
python gen_images_dataloader_with_render.py --network ./data/network-snapshot-025000.pkl --seeds 0 --outdir ./outdir
```

Different clothing types and colors can be combined to generate various geometries and textures. This can be achieved by examining the inference code. 

If one wishes to use the pretrained model to generate new color samples, this can be done by first writing textual comments and then computing the CLIP features as mentioned in the paper. 

We already provide the pre-computed CLIP and BLIP features for the samples shown in the main paper and the video for a smooth starting point.

## LICENSE
To use this codebase please agree to the license agreement at the [project website](https://sculpt.is.tue.mpg.de/license.html). Questions related to licensing could be addressed to ps-licensing@tue.mpg.de

## Citation

Please cite our paper in case you use our data and/or code.

```
@inproceedings{SCULPT:CVPR:2024,
  title = {{SCULPT}: Shape-Conditioned Unpaired Learning of Pose-dependent Clothed and Textured Human Meshes},
  author = {Sanyal, Soubhik and Ghosh, Partha and Yang, Jinlong and Black, Michael J. and Thies, Justus and Bolkart, Timo},
  booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year = {2024},
}

```
