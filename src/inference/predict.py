import numpy 
from src.preprocessing import face_detector
import torch
import pathlib
import json
import cv2
from efficientnet_pytorch import EfficientNet

class InferenceModel(object):

    @classmethod
    def from_config(cls, config_path: str):

        try:
            config_dir = pathlib.Path(config_path)
            img_config = json.load(fp=config_dir)

        except(FileNotFoundError):
            raise RuntimeError("invalid inference configuration file path")

        try:
            input_image_size = img_config.get("input_image_size")
            encoder_image_size = img_config.get("encoder_image_size")

            min_face_size = img_config.get("min_face_size", 160)
            inference_device = img_config.get("inference_device", "cpu")
            encoder_name = img_config.get("encoder_name")

        except(KeyError):
            raise RuntimeError("missing on key parameters")

        cls._input_image_size = input_image_size
        cls._encoder_image_size = encoder_image_size
        cls._inference_device = torch.device(inference_device)

        cls._face_detector = face_detector.MTCNNFaceDetector(
            image_size=input_image_size,
            use_landmarks=False,
            keep_all_pred_faces=False,
            min_face_size=min_face_size,
            inf_device=cls._inference_device
        )
        cls._deepfake_classifier = EfficientNet.from_pretrained(encoder_name.lower())
        return cls()

    def predict(self, input_img: numpy.ndarray):

        resized_img = cv2.resize(
            input_img, 
            (self._input_image_size, self._input_image_size), 
            cv2.INTER_LINEAR 
        )

        face_boxes = numpy.asarray(self._face_detector.detect_faces(resized_img))
        face_boxes = numpy.round(face_boxes, decimals=0).astype(numpy.uint8)

        cropped_faces = [
            torch.from_numpy(resized_img[f[0]:f[1], f[2]:f[3]]).permute(2, 0, 1)
             for f in face_boxes
        ]

        device_faces = torch.stack(cropped_faces).to(self.inference_device)
        predicted_probs = self._deepfake_classifier.forward(inputs=device_faces).cpu()
        
        predicted_probs = torch.argmax(predicted_probs, dim=1, keepdim=False)
        predicted_labels = numpy.where(predicted_probs >= 0.5, 1, 0)

        del cropped_faces 
        del device_faces
        del faces 
        del resized_img 

        return predicted_labels
        




