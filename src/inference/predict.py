import numpy 
from src.preprocessing import face_detector
import torch
import pathlib
import json
import cv2
from src.training.classifiers import classifiers

class InferenceModel(object):

    @classmethod
    def from_config(cls, config_path: str):

        try:
            config_dir = pathlib.Path(config_path)
            inf_config = json.load(fp=config_dir)

        except(FileNotFoundError):
            raise RuntimeError("invalid inference configuration file path")

        try:
            input_image_size = inf_config.get("input_image_size") # input image size, acceptable by MTCNN
            input_channels = inf_config.get("input_image_channels") # number of channels on the image for inference
            encoder_image_size = inf_config.get("encoder_image_size") # image size, acceptable by the encoder

            encoder_name = inf_config.get("encoder_name") # should be from 'classifiers.encoder_params'

            min_face_size = inf_config.get("min_face_size", 160) # minimum size of the face to detect
            inference_device = inf_config.get("inference_device", "cpu") # device for network inferencing
            net_weights_path = inf_config.get("network_weights_path") # path to the network weights

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
        try:
            network_config = torch.load(f=net_weights_path)

            cls._deepfake_classifier = classifiers.DeepfakeClassifierSRM(
                input_channels=input_channels,
                encoder_name=encoder_name,
                num_classes=2
            ).load_state_dict(
                state_dict=network_config['network']
            )
        except(FileNotFoundError):
            raise SystemExit("invalid path. failed to load network weights")
        
        except(KeyError):
            raise SystemExit("""failed to load network from configuration file. 
            'network' key does not exist in the config.""")

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

        predicted_probs = self._deepfake_classifier.to(
        self._inference_device).forward(inputs=device_faces).cpu()
        
        pred_faces_probs = torch.argmax(predicted_probs, dim=1, keepdim=False)
        output_preds = []

        for face_idx in range(len(device_faces)):
            output_preds.append(
                {
                    'face_coords': cropped_faces[face_idx],
                    'real_prob': pred_faces_probs[face_idx][0],
                    'fake_prob': pred_faces_probs[face_idx][1],
                }
            )

        del cropped_faces 
        del device_faces
        del faces 
        del resized_img 

        return output_preds
        

