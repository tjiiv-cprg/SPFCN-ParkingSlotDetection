# SPFCN: Select and Prune the Fully Convolutional Networks for Real-time Parking Slot Detection

The pytorch implement of the real-time parking slot detection method SPFCN.

Paper link: https://arxiv.org/abs/2003.11337

## Table of Content
 - [Abstract](#Abstract)
 - [Usage](#Usage)
 - [Performance](#Performance)
 - [Demo](#Demo)

## Abstract
For passenger cars equipped with automatic parking function, convolutional neural networks(CNN) are employed to detect parking slots on the panoramic surround view, which is an overhead image synthesized by four calibrated fish-eye images, The accuracy is obtained at the price of low speed or expensive computation equipments, which are sensitive for many car manufacturers. In this paper, the same accuracy is challenged by the proposed parking slot detector, which leverages deep convolutional networks for the faster speed and smaller model while keep the accuracy by simultaneously training and pruning it. To achieve the optimal trade-off, we developed a strategy to select the best receptive fields and prune the redundant channels automatically during training. The proposed model is capable of jointly detecting corners and line features of parking slots while running efficiently in real time on average CPU. Even without any specific computing devices, the model outperforms existing counterparts, at a frame rate of about 30 FPS on a 2.3 GHz CPU core, getting parking slot corner localization error of 1.51±2.14 cm (std. err.) and slot detection accuracy of 98%, generally satisfying the requirements in both speed and accuracy on on-board mobile terminals.

## Usage
Detailed instructions will be given soon.

## Performance
The training and test data set is https://cslinzhang.github.io/deepps/
| Method | Allowable deviation/cm | Parameter size/MB | Precision | Recall |
| :------: | :------: | :------:| :------: | :------: |
| PSD\_L| 16 | 8.38 | 0.9855 | 0.8464 |
| DeepPS| 16 | 255 | **0.9954** | **0.9889** |
| SPFCN(ours)| 6 |**2.39** | 0.9801 | 0.9731 |

## Demo
![image](https://github.com/LoyalBlanc/SPFCN-ParkingSlotDetection/blob/master/Demo/SPFCN-5s.gif)

A one-minute demo video can be seen in Demo/IV2020_SPFCN.mp4


