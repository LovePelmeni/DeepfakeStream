import numpy 

def match_fake_and_orig_faces(
    input_img: numpy.ndarray, 
    input_img2: numpy.ndarray
):
    """
    Function matches 2 different
    images using SIFT detector
    to find the degree of similarity 
    between them
    NOTE:
        when we extract original and deepfaked
        video frames, we have to match the original
        and deepfaked faces somehow, that's where
        SIFT and SSIM comes into play, as they provide
        the way of measuring similarity between 2 images.
    """
    pass