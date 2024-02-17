from calendar import c
from re import S
import numpy 
from src.preprocessing import face_detector
import torch
import pathlib
import json
import cv2
import glob
import os
import logging
import onnxruntime 

logger = logging.getLogger("model_loading_logger")
handler = logging.FileHandler(filename="model_loading_logs.log")
logger.addHandler(handler)

class InferenceModel(object):

    @classmethod
    def from_config(cls, config_path: str):

        try:
            config_dir = pathlib.Path(config_path)
            inf_config = json.load(fp=open(config_dir, mode='r'))

        except(FileNotFoundError) as err:
            logger.error(err)
            raise RuntimeError("invalid inference configuration file path")

        except(UnicodeDecodeError, json.decoder.JSONDecodeError) as dec_err:
            logger.error(dec_err)
            raise RuntimeError("Failed to parse inference configuration file.")

        try:
            encoder_image_size = inf_config.get("encoder_image_size") # image size, acceptable by the encoder
            min_face_size = inf_config.get("min_face_size", 160) # minimum size of the face to detect
            face_margin_size = inf_config.get("face_margin_size", 0) # margin to apply to face crop after 
            inference_device = inf_config.get("inference_device", "cpu") # device for network inferencing
            net_state_path = pathlib.Path(inf_config.get("network_weights_path")) # path to the network weights

        except(KeyError):
            raise RuntimeError("missing on key parameters")

        cls._encoder_image_size: int = int(encoder_image_size)
        cls._inference_device = torch.device(inference_device)

        cls._face_detector = face_detector.MTCNNFaceDetector(
            use_landmarks=False,
            keep_all_pred_faces=False,
            min_face_size=min_face_size,
            inf_device=cls._inference_device,
            margin=face_margin_size # size in pixels
        )
        
        try:
            state_name = os.path.basename(net_state_path)
            parent_dir = net_state_path.parent

            network_state_path = glob.glob(
                pathname="**/%s" % os.path.join(parent_dir.suffix, state_name),
                root_dir="weights"
            )
            
            if not len(network_state_path):
                raise FileNotFoundError

            net_path = os.path.join("weights", network_state_path[-1])
            
            cls.classifier_session = onnxruntime.InferenceSession(path_or_bytes=net_path)
            cls.inputs = cls.classifier_session.get_inputs()
            cls.run_options = onnxruntime.RunOptions()

        except(FileNotFoundError) as err:
            logger.error(err)
            raise SystemExit("invalid path. failed to load network weights")
        
        except(KeyError) as err:
            raise err

        return cls()

    def predict(self, input_img: numpy.ndarray):

        face_boxes, _ = self._face_detector.detect_faces(input_img)

        if len(face_boxes) == 0:
            return {}

        height, width = input_img.shape[:-1]
    
        cropped_faces = [
            input_img[
                min(max(round(f[1]),0),height):min(max(round(f[3]),0),height), 
                min(max(round(f[0]), 0),width):min(max(round(f[2]), 0),width)
            ]
            for f in face_boxes
        ]

        resized_faces = [
            cv2.resize(input_face, 
            (self._encoder_image_size, self._encoder_image_size),
            cv2.INTER_LINEAR)
            for input_face in cropped_faces
        ]

        device_faces = [
            torch.from_numpy(face).permute(2, 0, 1).float().unsqueeze(0).to(self._inference_device)
            for face in resized_faces
        ]

        input_names = self.classifier_session.get_inputs()

        predicted_probs = []

        for face in range(len(device_faces)):
            
            sample_probs = self.classifier_session.run(
                output_names=None, 
                input_feed={input_names[0].name: device_faces[face].numpy()}
            )

            predicted_probs.extend(sample_probs[-1].tolist())
        
        pred_faces_probs = numpy.amax(predicted_probs, axis=1, keepdims=False)
        pred_faces_labels = numpy.where(pred_faces_probs >= 0.5, 1, 0)

        output_preds = []

        for face_idx in range(len(device_faces)):
            output_preds.append(
                {
                    'face_coords': face_boxes[face_idx],
                    'label': int(pred_faces_labels[face_idx]),
                    'probability': pred_faces_probs[face_idx]
                }
            )

        del cropped_faces 
        del device_faces
        del resized_faces
        del input_img

        return output_preds
        

