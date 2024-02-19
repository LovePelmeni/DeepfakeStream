import numpy
import cv2
import dlib
import typing


def blackout_convex_hull(
    input_img: numpy.ndarray,
    face_boxes: typing.List[list]
):
    """
    Leverages Graham-Scan algorithm for 
    blacking out partial face zones:
    mouth, nose, eyes

    Parameters:
    ----------
        - input_img - (numpy.ndarray) - image for processing
        - landmarks - (list) - human face landmarks

    NOTE:
        - it is recommended to add small padding or margin
        to the input image, containing face, to make it appear
        in a full shape on the image
    """
    predictor = dlib.shape_predictor(
        "./face_shape_predictor/shape_predictor_68_face_landmarks.dat")

    for face in face_boxes:
        # obtaining 68 landmark points on the face
        predicted_landmarks = predictor(input_img, face)
        dots = [
            (
                predicted_landmarks.part(idx).x,
                predicted_landmarks.part(idx).y
            )
            for idx in range(len(predicted_landmarks))
        ]
