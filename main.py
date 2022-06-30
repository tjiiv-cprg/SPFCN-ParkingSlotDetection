import time
import cv2
import torch
import dill

from SPFCN_Light.slot_detector import Detector
from SPFCN import slot_network_training, slot_network_testing
from SPFCN.model import SlotDetector

if __name__ == "__main__":

    ### ORIGINAL VERSION ###
    # Train model
    slot_network_training(data_num=10, batch_size=12, valid_data_num=1500, valid_batch_size=32, epoch=10, input_res=224, device_id=0, num_workers=0)
    
    # Test model 
    params_path = './parameters/merge_bn_epoch10_loss4.pkl'
    slot_network_testing(parameter_path=params_path, data_num=1500, batch_size=50, input_res=224, device_id=0,  num_workers=0)

    # Load detector
    detector = SlotDetector(device_id=0, dim_encoder=[32, 44, 64, 92, 128], parameter_path=params_path)

    # Visualize the merge image with result
    current_frame = cv2.imread("demo.jpg")
    inference_image = cv2.resize(current_frame, (224, 224))
    inference_result = detector(inference_image)

    resolution = current_frame.shape[0]
    for detect_result in inference_result:
        pt0 = (int(detect_result[0][0] * resolution / 224), int(detect_result[0][1] * resolution / 224))
        pt1 = (int(detect_result[1][0] * resolution / 224), int(detect_result[1][1] * resolution / 224))
        pt2 = (int(detect_result[2][0] * resolution / 224), int(detect_result[2][1] * resolution / 224))
        pt3 = (int(detect_result[3][0] * resolution / 224), int(detect_result[3][1] * resolution / 224))
        cv2.line(current_frame, pt0, pt1, (0, 255, 0), thickness=2)
        cv2.line(current_frame, pt0, pt3, (0, 0, 255), thickness=2)
        cv2.line(current_frame, pt1, pt2, (0, 0, 255), thickness=2)
        cv2.line(current_frame, pt2, pt3, (0, 0, 255), thickness=2)
    cv2.imwrite("result.jpg", current_frame)


    ### LIGHT VERSION ###
    detector = Detector("./SPFCN_Light/stable_parameter_0914.pkl", device_id=-1)

    for frame_index in range(1000):
        tic = time.time()
        inference_image = cv2.cvtColor(cv2.resize(current_frame, (224, 224)), cv2.COLOR_BGR2GRAY)
        inference_result = detector(inference_image)
        toc = time.time()
        time_span = toc - tic
        infer_fps = 1 / (time_span + 1e-5)
        print("Frame:{:d}, Time used:{:.3f}, FPS:{:.3f}".format(frame_index, time_span * 1000, infer_fps), end='\r')

    resolution = current_frame.shape[0]
    for detect_result in inference_result:
        pt0 = (int(detect_result[0][0] * resolution / 224), int(detect_result[0][1] * resolution / 224))
        pt1 = (int(detect_result[1][0] * resolution / 224), int(detect_result[1][1] * resolution / 224))
        pt2 = (int(detect_result[2][0] * resolution / 224), int(detect_result[2][1] * resolution / 224))
        pt3 = (int(detect_result[3][0] * resolution / 224), int(detect_result[3][1] * resolution / 224))
        cv2.line(current_frame, pt0, pt1, (0, 255, 0), thickness=2)
        cv2.line(current_frame, pt0, pt2, (0, 0, 255), thickness=2)
        cv2.line(current_frame, pt1, pt3, (0, 0, 255), thickness=2)
        cv2.line(current_frame, pt2, pt3, (0, 0, 255), thickness=2)
    cv2.putText(current_frame, "%.2f fps" % infer_fps, (30, 30), cv2.FONT_HERSHEY_COMPLEX, 1.0, (0, 0, 255))
    cv2.imwrite("result.jpg", current_frame)
