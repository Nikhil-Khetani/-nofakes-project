import global_classifier
import argparse
import yoloV3.yolo_custom_detection.copy_yolo_object_detection as yolo
#path to the drn model for classifier
classifier_model_path = 'utils/dlib_face_detector/mmod_human_face_detector.dat'

def undo_warping(image):
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_path", required=True, help="the model input")
    parser.add_argument(
        "--gpu_id", default='0', help="the id of the gpu to run model on")
    parser.add_argument(
        "--no_crop",
        action="store_true",
        help="do not use a face detector, instead run on the full input image")
    args = parser.parse_args()

    image_path = args.input_path

    image= yolo.get_object(image_path)


    classifier_model = global_classifier.load_classifier(classifier_model_path, args.gpu_id)
    prob = global_classifier.classify_fake(classifier_model, image_path)




    if prob>0.5:
        print("This image is modified with probability {}".format(prob))
        undo_warping(image)
    else:
        print("This image is unmodified with probability {}".format(1-prob))




