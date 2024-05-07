# SCULPT_release (CVPR 2024)
The official codebase for SCULPT: Shape-Conditioned Unpaired Learning of Pose-dependent Clothed and Textured Human Meshes

**[Project Website](https://sculpt.is.tue.mpg.de)** | **[Dataset Download](https://sculpt.is.tue.mpg.de/download.php)** | **[Arxiv Paper](https://arxiv.org/pdf/2308.10638v2)**


## Training


First clone the github repo.

```
git clone https://github.com/soubhiksanyal/SCULPT_release.git
cd SCULPT_release
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


## Inference 

First, create a data folder. Next, download and extract all the data from the project website and place them into the data folder. Then, run the following command to generate the meshes and renderings used in the main paper and the video. 

```
python gen_images_dataloader_with_render.py --network ./data/network-snapshot-025000.pkl --seeds 0 --outdir ./outdir
```

Different clothing types and colors can be combined to generate various geometries and textures. This can be achieved by examining the inference code. 

If one wishes to use the pretrained model to generate new color samples, this can be done by first writing textual comments and then computing the CLIP features as mentioned in the paper. 

We already provide the pre-computed CLIP and BLIP features for the samples shown in the main paper and the video for a smooth starting point.

## LICENSE
Please check the license on the [project website](https://sculpt.is.tue.mpg.de/license.html). Questions related to licensing could be addressed to ps-licensing@tue.mpg.de
