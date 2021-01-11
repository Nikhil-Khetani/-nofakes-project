# -nofakes-project
this is the official repository for the #nofakes project provided by the University of Warwick AI society. We are building up on the classifier created by the adobe research team: https://github.com/PeterWang512/FALdetector. The aim of this project is to detect photoshopped images done to the body, not just human faces. The Train_YoloV3.ipynb allows us to train the yolo algorithm on custom labelled images to localize a human body. The model_trial.py in the networks folder uses a dilated residual network from the FAL repository, it is trained as a binary classifier to recognise whether an image is photoshopped or not. If it has been photoshopped then the yolo algorithm will be able to localize the body. All the necessary code has been uploaded to this GitHub repository but please see the references.docx for the references.

# Main results:

