# import required libraries
import csv

import cv2
import dlib
import numpy as np

# Define global paths for the models
FACE_REC_MODEL_PATH = './models/dlib_face_recognition_resnet_model_v1.dat'
PREDICTOR_PATH = './models/shape_predictor_5_face_landmarks.dat'

DESCRIPTORS_FILE_PATH = './users_descriptors.csv'

# Initialize the model objects
face_detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor(PREDICTOR_PATH)
face_recognition_model = dlib.face_recognition_model_v1(FACE_REC_MODEL_PATH)

CONFIDENCE_RATIO = 0.5  # Recognition confidence ratio
USER_NAME = "UNKNOWN"


def main():
    cap = cv2.VideoCapture(0)

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        frame = face_recognize(frame)
        if frame is not None:
            # Display the resulting frame
            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


def face_recognize(img):
    detector = face_detector(img, 1)

    val = CONFIDENCE_RATIO
    user_name = USER_NAME

    # If a face is detected
    for _, dimensions in enumerate(detector):
        shape = shape_predictor(img, dimensions)
        face_descriptor = face_recognition_model.compute_face_descriptor(img, shape)

        with open(DESCRIPTORS_FILE_PATH) as csv_file:
            # Read the file as a csv object
            reader = csv.DictReader(csv_file)

            # loop through rows
            for row in reader:
                j = np.asarray(row['descriptor'].split('\n'), dtype='float32')
                label = row['user_name']

                # Compute the deference between the descriptor of the detected face
                # and the descriptor in the csv file
                difference = np.linalg.norm(face_descriptor - j)

                # if the difference if less than the CONFIDENCE_RATIO
                if difference < CONFIDENCE_RATIO:
                    val = difference
                    user_name = label

        draw_shape(img, dimensions, user_name, val)

        return img


def draw_shape(img, dimensions, user_name, ratio):
    color = (0, 255, 0)  # Green color in BGR
    thickness = 2  # Line thickness of 2 px

    # Using cv2.rectangle() method
    cv2.rectangle(img,
                  (dimensions.left(), dimensions.top()),
                  (dimensions.right(), dimensions.bottom()),
                  color, thickness)

    # Draw a label with a name below the face
    cv2.rectangle(img,
                  (dimensions.left(), dimensions.bottom() - 35),
                  (dimensions.right(), dimensions.bottom()),
                  color, cv2.FILLED)

    font = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(img, user_name + ' ' + str(ratio),
                (dimensions.left() + 6, dimensions.bottom() - 6),
                font, 1.0,
                (255, 255, 255), 1)


if __name__ == '__main__':
    main()
