from torch.utils import data 
import abc
import torch
from facenet_pytorch import MTCNN
import typing
import cv2
from collections import OrderedDict

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

class VideoFaceDetector(abc.ABC):

    @abc.abstractproperty
    def batch_size(self):
        pass

    @abc.abstractmethod
    def detect_faces(self, input_img: torch.Tensor):
        pass

class MTCNNFaceDetector(object):
    """
    Face Detector CNN Network, based on
    MTCNN (Multi-task Cascaded Convolutional Neural Network) model.

    Parameters:
    -----------
    image_size (int) - expected input size of the image (ideally aligns with the MTCNN training size)
    use_landmarks (bool) - specifies, whether network needs to generate landmarks (segmentation masks) for output faces
    keep_all_pred_faces (bool) - option enables model to accept and return all possible face predictions (even weak ones) without using any threshold values.
    inf_device ('cpu' or 'cuda') - device for perfoming inference, can be separate GPU node or CPU core.
    min_face_size - (int) - minimum size of the human face to detect on the image
    """
    def __init__(self, 
        image_size: int, 
        use_landmarks: bool = False,
        keep_all_pred_faces: bool = False,
        min_face_size: int = 160,
        inf_device: typing.Literal['cuda', 'cpu'] = 'cpu'):

        self.detector = MTCNN(
            post_process=True, # disable image normalization after detection
            margin=0.6 * image_size, # padding of area for bounding boxes
            min_face_size=min_face_size,
            device=inf_device,
            keep_all=keep_all_pred_faces,
            thresholds=[0.95, 0.95, 0.95] # NMS probs for P-Net, R-Net and O-net
        )
        self.use_landmarks = use_landmarks

    def detect_faces(self, input_img: torch.Tensor):
        face_boxes, *_, face_landmarks = self.detector.detect(
            input_img, 
            landmarks=self.use_landmarks
        )
        return [box.tolist() for box in face_boxes], [face.tolist() for face in face_landmarks]

class VideoFaceDataset(data.Dataset):
    """
    Dataset for processing deepfake videos
    and extracting human faces from the 
    image scenes
    """
    def __init__(self, video_paths: typing.List):
        self.video_paths = video_paths

    def __len__(self):
        return len(self.video_paths)
    
    def __getitem__(self, idx: int):

        video_path = self.video_paths[idx]
        video_buffer = cv2.VideoCapture(filename=video_path)
        
        frame_num = int(video_buffer.get(cv2.CAP_PROP_FRAME_COUNT))
        frames = []

        for idx in range(frame_num): # pick each 32-th video frame

            success, frame = video_buffer.read()

            if not success: 
                continue 

            if len(frame.shape) == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            frames.append(frame)
        return frames

