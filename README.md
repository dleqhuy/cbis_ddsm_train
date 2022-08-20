# cbis_ddsm_train

![Fig1](https://raw.githubusercontent.com/lishen/end2end-all-conv/master/ddsm_train/Fig-1%20patch%20to%20whole%20image%20conv.jpg "Convert conv net from patch to whole image")

[https://github.com/lishen/end2end-all-conv/](https://github.com/lishen/end2end-all-conv/)


## Whole image model downloads
Here is a table for model acc:

1. Set learning rate to 10−4, and train the newly added top layers for 30 epochs. 
2. Set learning rate to 10−5, and train all layers for 20 epochs for a total of 50 epochs.

| Database  | Patch Classifier  | Top Layers (two blocks)  | Accuracy  |
|---|---|---|---|
| CBIS-DDSM  | Resnet50  | 512x1  | 0.73529  |
| CBIS-DDSM  | VGG16  | 512x1  | 0.78676  |

## Patch classifier model

The 3-stage training strategy on the patch set was as follows:

1. Set learning rate to 10−3 and train the last layer for 3 epochs. 
2. Set learning rate to 10−4, unfreeze the top layers and train for 10 epochs, where the top layer number is set to 46 for Resnet50 and 11 for VGG16. 
3. Set learning rate to 10−5, unfreeze all layers and train for 37 epochs for a total of 50 epochs.

Here is a table for model acc:

| Model  | Train Set | Accuracy |
|---|---|---|
| Resnet50  | S10  | 0.83418  |
| VGG16  | S10  | 0.80547  |
