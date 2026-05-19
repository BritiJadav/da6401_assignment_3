# Building a Complete Visual Perception Pipeline


[Github link](https://github.com/BritiJadav/da6401_assignment_2)


[Wandb Report Link](https://wandb.ai/britisundarghatak100-iit-madras/da6401-assignment-2/reports/da6401_Assignment_2--VmlldzoxNjQ4ODM0MA?accessToken=lk9wwi6vjezfpggd25ioowbw54n3vvl3qnj4ujirpa6wqh6uno6va27a8k2ovwfg)


This project implements a unified multi-task visual perception pipeline using a shared VGG11 encoder backbone trained on the Oxford-IIIT Pet dataset. The pipeline simultaneously performs three vision tasks: breed classification across 37 categories, bounding box localization using a regression head with IoU loss, and pixel-wise semantic segmentation using a U-Net style decoder with skip connections. Key architectural decisions include custom inverted dropout for regularization, batch normalization for training stability, and a transfer learning strategy where the shared encoder is initialized from a pretrained classifier checkpoint. All experiments including training curves, activation distributions, feature map visualizations, segmentation masks, and bounding box predictions are tracked and logged using Weights & Biases (W&B) for comprehensive experiment management.
