# vertebrae_cls Documentation

## Description

The goal of this project is to create a 3D classifier for spinal injuries. 

To achieve this, the following steps are performed:
- Data is sourced from real-world DICOM files.
- The data is preprocessed and adapted to fit the requirements of the classifier.
- Vertebrae are segmented using the MONAI WholeBody CT segmentation model, which is used solely for the purpose of extracting individual vertebrae from the scans.
- The segmented vertebrae are then used to train and evaluate a custom 3D classification model for spinal injuries.

This pipeline ensures that the data is properly prepared for the development of a robust and accurate 3D classifier for spinal injury detection.