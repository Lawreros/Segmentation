# Segmentation
This repository is meant to contain all of my experimentation with (and replication of) the different image segmentation papers and concepts that I come across. Each paper will be given its own folder, with a `README.md` for each which further explains its contents. Please see the end of this `README` for the License Agreement.

## General Repo-Specific Custom Code
The `gen_utils` directory contains the classes and functions that I have created that perform functions that are used across the different segmentation techniques. These are mainly related to the loading/processing/sampling of input `.nii`, `.png`, and `.dcm` files and may be imported in the different notebooks.

## 1. Attention Enriched Deep Learning Model for Brest Tumor Segmentation in Ultrasound Images:
```
@ARTICLE{Vakanski2020,
    title={Attention-Enriched Deep Learning Model for Breast Tumor Segmentation in Ultrasound Images},
    author={Vakanski, A. and Xian, M. and Freer, P. E.},
    journal={Ultrasound in Medicine & Biology}, 
    year={2020},
    month={Sep.},
    volume={46},
    number={10}
    pages={2819-2833},
    doi={https://doi.org/10.1016/j.ultrasmedbio.2020.06.015}
}
```
Contained in the `BTSU` directory. The citation for the paper is found above, with the official github repository found here: https://github.com/avakanski/Attention-Enriched-DL-Model-for-Breast-Tumor-Segmentation which uses keras (instead of the pytorch version I will be creating)
