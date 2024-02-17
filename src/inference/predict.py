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

        except(FileNotFoundError):
            raise SystemExit("invalid path. failed to load network weights")
        
        except(KeyError) as err:
            raise err

        return cls()

    def predict(self, input_img: numpy.ndarray):

        resized_img = cv2.resize(
            input_img, 
            (self._encoder_image_size, self._encoder_image_size), 
            cv2.INTER_LINEAR 
        )

        face_boxes = numpy.asarray(self._face_detector.detect_faces(resized_img))
        face_boxes = numpy.round(face_boxes, decimals=0).astype(numpy.uint8)

        cropped_faces = [
            torch.from_numpy(resized_img[f[0]:f[1], f[2]:f[3]]).permute(2, 0, 1)
             for f in face_boxes
        ]

        device_faces = [face.to(self.inference_device) for face in cropped_faces]

        inputs = {
            self.inputs[idx].name: device_faces[idx]
            for idx in range(len(self.inputs))
        }

        predicted_probs = self.classifier_session.run(
            output_names=None, 
            input_feed=inputs,
            run_options=self.run_options
        )
        
        pred_faces_probs = torch.argmax(predicted_probs, dim=1, keepdim=False)
        pred_faces_labels = numpy.where(predicted_probs >= 0.5, 1, 0)
        output_preds = []

        for face_idx in range(len(device_faces)):
            output_preds.append(
                {
                    'face_coords': cropped_faces[face_idx],
                    'label': pred_faces_labels[face_idx],
                    'probability': pred_faces_probs[face_idx]
                }
            )

        del cropped_faces 
        del device_faces
        del faces 
        del resized_img 

        return output_preds
        

