# EE 210A Project: Binary classification of normal vs Alzheimer’s Disease using 3D CNN #

Several convolutional neural networks(CNNs) were used to attempt to classify Alzheimer's existence in patients. 
The data pre-processing classes/functions are described in the section *Helper Files* along with a unit test file for debugging purposes. 
The actual CNN architecture implementations are the network_model files and are described in *Scripts*.

* Quick visual inspection of the original cropped data can be seen here (coronal view only):
http://users.bmap.ucla.edu/~yeunkim/OASIS/

* Quick visual inspection of the cropped data set for our CNN can be seen here:
http://users.bmap.ucla.edu/~yeunkim/OASIS/cropped/Original_cropped_pkl2/


## Helper Files: ##
* *Load_data(labels, MRI_scans, Jacobian_scans,(sizeX,sizeY,sizeZ))*: Partitions training and test data labels/IDs. Called from test and network_model to load in MRI scans. 

* *Preprocess(data, smoothing factor)*: Performs smoothing and per image/per dataset normalization. Called from test and network_model to preprocess the MRI scans. 

* *Metadata(panda_dataframe,desired_features,data)*: Extracts the svm features from the metadata collected in the study. Called from network_models.

* *Test*: Unit tests the functions(helper files).


## Scripts: ##

* *3D_Network_model_1.py* : Runs the 3D CNN to classify Alzheimer’s (refer to slide number 5 in pptx).

* *3D_Network_model_2.py* : Runs the 3D CNN to classify Alzheimer’s (refer to slide number 6 in pptx).

* *2D_Network_model.py*: Runs the 2D CNN architecture to classify Alzheimer’s. This architecture combines CNN extracted features with SVM classifier.

## Miscellaneous scripts: ##
* *convert.sh* : Converts the ANALYZE files into NIfTI format.

* *svreg_18a.sh* : Performs surface-constrained volumetric registration of the subject image data to an atlas using BrainSuite v18a (www.brainsuite.org).

# Usage: #
To run the CNN models, the following variables need to be set at the beginning of the scripts:

```python
fn = "/home/yklocal/Downloads/new_Y_Data.csv"
dirimg = "/home/yklocal/Downloads/Original_cropped_pkl2/"
dirjac = "/home/yklocal/Downloads/Jac_cropped/"
```

Where fn indicates the CSV file with the subject IDs, binary classification labels, and metadata. 
dirimg is the directory that has the MR volumetric data.
dirjac is the directory with the jacobian determinant data.

Then the script can be executed.

Github repo: https://github.com/yeunkim/EE210AProject.git
