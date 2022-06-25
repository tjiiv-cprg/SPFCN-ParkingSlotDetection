import time
import cv2
import torch
import dill

from SPFCN_Light.slot_detector import Detector
from SPFCN import slot_network_training, slot_network_testing

if __name__ == "__main__":

    # auto train
    slot_network_training(data_num=6535, batch_size=10, epoch=10, input_res=224, device_id=0, num_workers=0)

    # auto test 
    model_path = './SPFCN/'
    slot_network_testing(model_path, device_id=0)

    # Load detector
    detector = LoadDetector()

    # Visualize the merge image with result
    current_frame = cv2.imread("demo.jpg")
    inference_image = cv2.resize(current_frame, (224, 224))
    inference_result = detector(inference_image)



    ### LIGHT VERSION ###
    # Read image
    current_frame = cv2.imread("demo.jpg")

    # Initial model
    detector = Detector("./SPFCN_Light/stable_parameter_0914.pkl", device_id=-1)

    # Start the detection
    for frame_index in range(1000):
        # Get the result
        tic = time.time()
        inference_image = cv2.cvtColor(cv2.resize(current_frame, (224, 224)), cv2.COLOR_BGR2GRAY)
        inference_result = detector(inference_image)
        toc = time.time()
        time_span = toc - tic
        infer_fps = 1 / (time_span + 1e-5)
        print("Frame:{:d}, Time used:{:.3f}, FPS:{:.3f}".format(frame_index, time_span * 1000, infer_fps), end='\r')

    # Visualize the merge image with result
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
