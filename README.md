# -Covidx-IMICS-Lung-Segmentation
# COVIDx-iMICS Lung Segmentation Dataset
**Update 08/11/2020: Released new dataset with over 14000 CXR images containing 473 COVID-19 train samples. Train and Test data are kept the same as provided bt this script https://github.com/lindawangg/COVID-Net/blob/master/docs/COVIDx.md
** If you are using this dataset, please cite this paper;  https://arxiv.org/abs/2010.06418

# HOW TO GENERATE SEGMENTED COVIDX DATASET:

  1) Please download the original COVIDx dataset by following https://github.com/ieee8023/covid-chestxray-dataset. 
  2) Save the Train and Test data under folders train and test
  3) Run the test.py file which uses our pre-trained weighits to generate masks for COVIDx dataset using a pre-trained unet model
  4) Make sure to change the file paths in test.py in accordance to your directory structure

COVIDX iMICS lung segmentation dataset can be downloaded from the link below;
https://utoronto-my.sharepoint.com/:f:/g/personal/sam_motamed_mail_utoronto_ca/Es8e4xREsxdFvwAU8g83UhgBCIl2BS32Nz1MaO-_vRDz6Q?e=w9qmi4

The current COVIDx dataset is constructed by the following open source chest radiography datasets:
* https://github.com/ieee8023/covid-chestxray-dataset
* https://github.com/agchung/Figure1-COVID-chestxray-dataset
* https://github.com/agchung/Actualmed-COVID-chestxray-dataset
* https://www.kaggle.com/tawsifurrahman/covid19-radiography-database
* https://www.kaggle.com/c/rsna-pneumonia-detection-challenge (which came from: https://nihcc.app.box.com/v/ChestXray-NIHCC )


