# SPFCN: Select and Prune the Fully Convolutional Networks for Real-time Parking Slot Detection

The pytorch implement of the real-time parking slot detection method SPFCN.

Paper link: https://arxiv.org/abs/2003.11337

## Table of Content
 - [Abstract](#Abstract)
 - [Usage](#Usage)
 - [Performance](#Performance)
 - [Demo](#Demo)

## Abstract
For vehicles equipped with the automatic parking system, the accuracy and speed of the parking slot detection are crucial. But the high accuracy is obtained at the price of low speed or expensive computation equipment, which are sensitive for many car manufacturers. In this paper, we proposed a detector using CNN(convolutional neural networks) for faster speed and smaller model size while keeps accuracy. To achieve the optimal balance, we developed a strategy to select the best receptive ﬁelds and prune the redundant channels automatically after each training epoch. The proposed model is capable of jointly detecting corners and line features of parking slots while running efﬁciently in real time on average processors. The model has a frame rate of about 30 FPS on a 2.3 GHz CPU core, yielding parking slot corner localization error of 1.51±2.14 cm (std. err.) and slot detection accuracy of 98%, generally satisfying the requirements in both speed and accuracy on onboard mobile terminals.

## Usage

1. You can set your data path in './SPFCN/dataset/__init__.py'.  

2. slot_network_training : A function that runs the network training code.

3. slot_network_testing : A function that runs the network testing code.

4. SlotDetector : A class that helps to return coordinate values ​​that can be used in an image based on the results of the network.


## Performance
The training and test data set is https://cslinzhang.github.io/deepps/
| Method | Allowable deviation/cm | Parameter size/MB | Precision | Recall |
| :------: | :------: | :------:| :------: | :------: |
| PSD\_L| 16 | 8.38 | 0.9855 | 0.8464 |
| DeepPS| 16 | 255 | **0.9954** | **0.9889** |
| SPFCN(ours)| 6 |**2.39** | 0.9801 | 0.9731 |

## Demo
![image](https://github.com/tjiiv-cprg/SPFCN-ParkingSlotDetection/blob/master/demo/SPFCN-5s.gif)

In the one-minute demo video, we show the results of the model's detection under a variety of conditions such as strong light interference, vehicle cornering and shaking. It can be considered that the location coordinates are well given to meet the needs of the actual vehicle.

Any player that can play MP4 will work, Windows Media Player recommended.


