"""
Find the bounding box of an object
===================================

This example shows how to extract the bounding box of the largest object

"""

import numpy as np
import cv2
import torch

from scipy import ndimage

def connected_component(im):
    # l=224
    # n=10
    # im = ndimage.gaussian_filter(im, sigma=l/(4.*n))

    mask = im > im.mean()
    # mask = im > 0

    label_im, nb_labels = ndimage.label(mask)

    # Find the largest connected component

    # sizes = ndimage.sum(mask, label_im, range(nb_labels + 1))

    # mask_size = sizes < 1000
    # remove_pixel = mask_size[label_im]
    # label_im[remove_pixel] = 0

    labels = np.unique(label_im)
    label_im = np.searchsorted(labels, label_im)
    # Now that we have only one connected component, extract it's bounding box
    object_slices = ndimage.find_objects(label_im)
    # Find the object with the largest area
    areas = [np.product([x.stop - x.start for x in slc]) for slc in object_slices]
    largest = object_slices[np.argmax(areas)]

    # for i in labels[1:]:
    #     slice_x, slice_y = ndimage.find_objects(label_im == i)[0]
    #     pdb.set_trace()
    #     boxAArea = (int(slice_x[1]) - int(slice_x[0]) + 1) * (int(slice_y[1]) - int(slice_y[0]) + 1)
    #     if boxAArea >= box_max:
    #         box_max = boxAArea
    #         j = i
    # slice_x, slice_y = ndimage.find_objects(label_im == j)[0]
    return [largest[0].start, largest[0].stop], [largest[1].start, largest[1].stop]

def returnCCAM_(feature_conv, weight_softmax, class_idx, reverse_idx, h_x, j, threshold, thr, function=None, num_classes=None):
    assert num_classes is not None, "num_classes is not provided"
    size_upsample = (224, 224)
    bz, nc, h, w = feature_conv.shape
    # print(bz, nc, h, w)
    output_cam = []
    # cam = weight_softmax[class_idx].dot(feature_conv.reshape((nc, h * w)))
    # cam1 = weight_softmax[reverse_idx].dot(feature_conv.reshape((nc, h * w)))
    
    cam = torch.matmul(weight_softmax[class_idx], feature_conv.reshape((nc, h * w)))
    cam1 = torch.matmul(weight_softmax[reverse_idx], feature_conv.reshape((nc, h * w)))

    if function == 't1b0':
        cam = cam.reshape(h, w)
        # cam = cam - np.min(cam)
        # cam_img = cam / np.max(cam)
        # cam_img = np.uint8(255 * cam_img)
        cam_img = cam

    elif function == 't1b1':
        cam = cam - cam1[-1]
        cam = cam.reshape(h, w)
        # cam = cam - np.min(cam)
        # cam_img = cam / np.max(cam)
        # cam_img = np.uint8(255 * cam_img)
        cam_img = cam

    elif function == 'linear':
        h_x = np.linspace(1, -1, num_classes)
        cam = np.matmul(h_x, cam1)
        cam = cam.reshape(h, w)
        # cam = cam - np.min(cam)
        # cam_img = cam / np.max(cam)
        # cam_img = np.uint8(255 * cam_img)
        cam_img = cam

    elif function == 'quadratic':
        h_x1 = np.linspace(1, 0, thr)
        h_x1 = h_x1 * h_x1
        h_x2 = np.linspace(0, -1, num_classes-thr)
        h_x2 = h_x2 * h_x2
        h_x2 = - h_x2
        h_x = np.concatenate([h_x1, h_x2])
        cam = torch.matmul(torch.tensor(h_x, device=cam1.device, dtype=cam1.dtype), cam1)#np.matmul(h_x, cam1)
        cam = cam.reshape(h, w)
        # cam = cam - np.min(cam)
        # cam_img = cam / np.max(cam)
        # cam_img = np.uint8(255 * cam_img)
        cam_img = cam

    else:
        raise ValueError('please select combinational function')
        print('please select combinational function')
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        cam_img = np.uint8(255 * cam_img)

    # output_cam.append(cv2.resize(cam_img, size_upsample))
    # cam = cv2.resize(cam, size_upsample)

    # bounding_box_thr = np.amax(cam) * threshold
    # cutting_image = np.where(cam > bounding_box_thr, cam, 0)
    # [slice_y, slice_x] = connected_component(cutting_image)
    # bounding_box = [slice_x[0], slice_y[0], slice_x[1], slice_y[1]]
    return cam_img#output_cam#, bounding_box