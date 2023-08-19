import cv2
import numpy as np
import matplotlib as plt


def skin_plot(img_add, iter):
    # load image, gt, prediction
    img = cv2.imread(img_add + str(iter) + '/img_' + str(iter) + '.png')
    gt = cv2.imread(img_add + str(iter) + '/gt_' + str(iter) + '.png')
    pred = cv2.imread(img_add + str(iter) + '/pred_' + str(iter) + '.png')
    
    # img = np.uint8(img)
    # gt = np.uint8(gt)
    # pred = np.uint8(pred)
    # convert mask and prediction to grayscale
    gt = cv2.cvtColor(gt, cv2.COLOR_RGB2GRAY)
    pred = cv2.cvtColor(pred, cv2.COLOR_RGB2GRAY)
    
    # extract boundaries from gt and pred
    gt_contours, gt_hierarchy = cv2.findContours(gt, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    pred_contours, pred_hierarchy = cv2.findContours(pred, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # draw the contours on the image
    for gt_cont in gt_contours:
        cv2.drawContours(img, [gt_cont], -1, (255,0,0), 1)
        
    for pred_cont in pred_contours:
        cv2.drawContours(img, [pred_cont], -1, (0,255,0), 1)
        
    # save the image
    cv2.imwrite(img_add + str(iter) + '/contour_pred_' + str(iter) + '.png', img)
    

if __name__ == '__main__':
    
    image_address = './output_vis_SwinUnet_2018/'
    
    for i in range(520):
        print(i+1)
        skin_plot(img_add=image_address, iter=i+1)