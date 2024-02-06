from torch.utils import data 
import abc
import torch
from facenet_pytorch import MTCNN
import typing
import cv2
import numpy.random
import logging 
import gc
import numpy

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

Logger = logging.getLogger("video_dataset_logger")


class VideoFaceDetector(abc.ABC):

    @abc.abstractmethod
    def detect_faces(self, input_img: torch.Tensor):
        pass

class MTCNNFaceDetector(VideoFaceDetector):
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
        inf_device: typing.Literal['cuda', 'cpu'] = 'cpu',
        i_conf_threshold: float = 0.85,
        r_conf_threshold: float = 0.95,
        o_conf_threshold: float = 0.95
    ):

        self.detector = MTCNN(
            post_process=True, # disable image normalization after detection
            margin=0.6 * image_size, # padding of area for bounding boxes
            min_face_size=min_face_size,
            device=inf_device,
            keep_all=keep_all_pred_faces,
            thresholds=[
                i_conf_threshold,
                r_conf_threshold,
                o_conf_threshold 
            ] # NMS probs for P-Net, R-Net and O-net
        )
        self.use_landmarks = use_landmarks

    def detect_faces(self, input_img: numpy.ndarray):

        face_landmarks = None 

        if self.use_landmarks == True:
            face_boxes, probs, face_landmarks = self.detector.detect(
                img=input_img, 
                landmarks=self.use_landmarks
            )
        else:
            face_boxes, probs = self.detector.detect(
                img=input_img, 
                landmarks=self.use_landmarks
            )

        if face_boxes is None or probs is None:
            return ([], [])

        face_boxes = [box.tolist() for box in face_boxes]
        
        print(face_boxes)
        print(face_landmarks)
        
        if self.use_landmarks: return face_boxes, probs, face_landmarks 
        return face_boxes, probs
class VideoFaceDataset(data.Dataset):
    """
    Dataset for processing deepfake videos
    and extracting human faces from the 
    image scenes

    Parameters:
    -----------
    
    video_paths - (list) - list of video paths 
    frames_per_vid - (int) - frames to extract from each video
    (number should be picked with thorough consideration, bigger number is recommended for short videos)
    """

    def __init__(self, 
        video_paths: typing.List, 
        video_labels: typing.List,
        frames_per_vid: float = 0.5,
    ):
        self.video_paths = video_paths
        self.video_labels = video_labels
        self.frames_per_vid = frames_per_vid

    def __len__(self):
        return len(self.video_paths)

    def extract_frames(self, video_path: str):
    
        try:
            video_buffer = cv2.VideoCapture(video_path)
            frame_num = int(video_buffer.get(cv2.CAP_PROP_FRAME_COUNT))
            
        except(UserWarning) as warn_err:
            Logger.warn(warn_err)

        except(FileNotFoundError, Exception):
            return []

        if frame_num == 0: return []

        frames_to_extract = numpy.random.choice(
            a=numpy.arange(int(frame_num)),
            size=max(int(self.frames_per_vid * frame_num), 1)
        )

        frames = []

        for idx in range(frame_num): # pick each 32-th video frame

            success = video_buffer.grab()

            if not success: 
                continue 
                
            del success

            _, frame = video_buffer.retrieve()

            if len(frame.shape) == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            if idx in frames_to_extract:
                frames.append(frame)
            else:
                del frame
                gc.collect()

        # closing video buffer after extracting frames 
        video_buffer.release()
        return frames
    
    def __getitem__(self, idx: int):
        video_path = self.video_paths[idx]
        video_label = self.video_labels[idx]
        frames = self.extract_frames(video_path)
        return video_label, frames




