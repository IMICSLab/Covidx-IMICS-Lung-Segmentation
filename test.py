                        #!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""

Sam Motamed

"""

from __future__ import division, print_function
from matplotlib import pyplot as plt
import numpy as np
print()
import matplotlib.pyplot as plt
from skimage.transform import resize
from keras.optimizers import Adam
from keras import backend as K
from models import actual_unet, simple_unet, N2
import os
import cv2
from statistics import mean, median
import matplotlib.gridspec as gridspec
import tensorflow as tf
import PIL
from skimage.transform import resize
from PIL import Image, ImageTk
def dice_coef(y_true, y_pred):

    tf.cast(y_pred, tf.int32)
    #tf.cast(y_pred, tf.float32)
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    # y_pred is softmax output of shape (num_samples, num_classes)
    # y_true is one hot encoding of target (shape= (num_samples, num_classes))
    intersect = K.sum(y_pred * y_true)

    denominator = K.sum(y_pred_f) + K.sum(y_true_f)
    dice_score = K.constant(2.) * intersect / (denominator)
    result = tf.keras.backend.switch(
    K.equal(denominator, 0.),
    0.,
    dice_score
)
    return result
    
def numpy_dice(y_true, y_pred):
    intersection = y_true*y_pred

    return ( 2. * intersection.sum())/ (np.sum(y_true) + np.sum(y_pred))
    


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

def make_plots(img, segm, segm_pred):
    n_cols=3
    n_rows = len(img)

    fig = plt.figure(figsize=[ 4*n_cols, int(4*n_rows)] )
    gs = gridspec.GridSpec( n_rows , n_cols )

    for mm in range( len(img) ):

        ax = fig.add_subplot(gs[n_cols*mm])
        ax.imshow(img[mm], cmap='gray' )
        ax.axis("image")  # remove axis
        ax.set_aspect(1)  # aspect ratio of 1
        if mm==0:
            ax.set_title('MRI image', fontsize=20)
        ax = fig.add_subplot(gs[n_cols*mm+1])
        ax.imshow(segm[mm], cmap='gray' )
        ax.axis("image")  # remove axis
        ax.set_aspect(1)  # aspect ratio of 1
        if mm==0:
            ax.set_title('True Mask', fontsize=20)

        ax = fig.add_subplot(gs[n_cols*mm+2])
        ax.imshow(segm_pred[mm], cmap='gray' )
        ax.axis("image")  # remove axis
        ax.set_aspect(1)  # aspect ratio of 1
        if mm==0:
            ax.set_title('Predicted Mask', fontsize=20)
    return fig

smooth = 1.
K.set_image_data_format('channels_last')  # TF dimension ordering in this code


def predict_mask(n_best=5, n_worst=5):
    if not os.path.isdir('./images'):
        os.mkdir('./images')
        
    if not os.path.isdir('./predicted_masks'):
        os.mkdir('./predicted_masks')
    img_rows = 256
    img_cols = 256
    model = N2(img_rows, img_cols)
    model.load_weights('./weights.h5')
    #model.load_weights('C:/Users/mshri/Desktop/MEAN_SHIFT_DWI/weights.h5')
    model.compile(optimizer=Adam(), loss=dice_coef_loss, metrics=[dice_coef, 'binary_accuracy'])

    kernel = np.ones((5,5),np.uint8)
    i = 0
    #print(model.metrics_names)
    for file in os.listdir('./test'):
        
        img = Image.open('./test/' + file).convert('L')
        img = np.asarray(img.resize((256, 256)))
        imgarr = np.array(img) / np.max(img) 
        new_X_test = imgarr.reshape(1,256,256, 1)
        y_pred = model.predict(new_X_test, verbose=1)
        y_pred = (y_pred > 0.1)
        y_pred =  (y_pred.astype(np.uint8))
        y_pred = cv2.morphologyEx(y_pred.reshape(256, 256), cv2.MORPH_CLOSE, kernel).reshape(256, 256)
        y_pred = cv2.morphologyEx(y_pred.reshape(256, 256), cv2.MORPH_OPEN, kernel).reshape(256, 256)
        result = Image.fromarray((y_pred * 255).astype(np.uint8))
        result = result.resize((1024, 1024))
        i += 1
        result.save('./test_preds/' + file)
    print(i)
        


if __name__=='__main__':
    predict_mask( )
