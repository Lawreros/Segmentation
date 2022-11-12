# Segmentation
This repository is meant to contain all of my experimentation with (and replication of) the different image segmentation papers and concepts that I come across. Each paper will be given its own folder, with a `README.md` for each which further explains its contents. Please see the end of this `README` for the License Agreement.

## General Repo-Specific Custom Code
The `gen_utils` directory contains the classes and functions that I have created that perform functions that are used across the different segmentation techniques. These are mainly related to the loading/processing/sampling of input `.nii`, `.png`, and `.dcm` files and may be imported in the different notebooks.

## 1. 3D U^2-Net: A 3D Universal U-Net for Multi-Domain Medical Image Segmentation:
```
@ARTICLE{Huang2019,
    title={3D U$^2$-Net: A 3D Universal U-Net for Multi-Domain Medical Image Segmentation},
    author={Huang, C. and Han, H. and Yao, Q. and Zhu, S. and Zhou, S. K.},
    journal={arXiv}, 
    year={2019},
    month={Sep.},
    doi={10.48550/arXiv.1909.06012}
}
```
Contained in the `u2net` directory. The citation for the paper is found above, with the official github repository found here: https://github.com/huangmozhilv/u2net_torch. The purpose of this implementation is both to improve my understanding of the use of U^2-nets, as well as provide an alternative learning example/documentation for others reading the source material.
