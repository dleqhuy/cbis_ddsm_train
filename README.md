# cbis_ddsm_train

![Fig1](https://raw.githubusercontent.com/lishen/end2end-all-conv/master/ddsm_train/Fig-1%20patch%20to%20whole%20image%20conv.jpg "Convert conv net from patch to whole image")

[https://github.com/lishen/end2end-all-conv/](https://github.com/lishen/end2end-all-conv/)


## Whole image model downloads
A few best whole image models are available for downloading at this Google Drive [folder](https://drive.google.com/drive/folders/0B1PVLadG_dCKV2pZem5MTjc1cHc?resourcekey=0-t4vtopuv27D9NnMC97w6hg&usp=sharing). YaroslavNet is the DM challenge top-performing team's [method](https://www.synapse.org/#!Synapse:syn9773040/wiki/426908). Here is a table for model AUCs:

| Database  | Patch Classifier  | Top Layers (two blocks)  | Accuracy  |
|---|---|---|---|
| CBIS-DDSM  | Resnet50  | 512x1  | 0.73529  |
| CBIS-DDSM  | VGG16  | 512x1  | 0.78676  |

## Patch classifier model downloads
Here is a table for model acc:

| Model  | Train Set | Accuracy |
|---|---|---|
| Resnet50  | S10  | 0.83418  |
| VGG16  | S10  | 0.80547  |
