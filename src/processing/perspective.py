import cv2
import numpy as np

class PerspectiveTransformation:
    """ This a class for transforming image between front view and top view

    Attributes:
        src (np.array): Coordinates of 4 source points
        dst (np.array): Coordinates of 4 destination points
        M (np.array): Matrix to transform image from front view to top view
        M_inv (np.array): Matrix to transform image from top view to front view
    """
    def __init__(self):
        """Init PerspectiveTransformation."""
        # These will be calculated when first image is processed
        self.src = None
        self.dst = None
        self.M = None
        self.M_inv = None

    def _calculate_transform(self, img_shape):
        """Calculate the perspective transform matrices based on image dimensions"""
        height, width = img_shape[:2]
        
        # Calculate source points as percentages of image dimensions
        self.src = np.float32([
            (width * 0.43, height * 0.64),     # top-left
            (width * 0.12, height),            # bottom-left
            (width * 0.94, height),            # bottom-right
            (width * 0.60, height * 0.64)      # top-right
        ])
        
        # Calculate destination points
        self.dst = np.float32([
            (width * 0.08, 0),                 # top-left
            (width * 0.08, height),            # bottom-left
            (width * 0.86, height),            # bottom-right
            (width * 0.86, 0)                  # top-right
        ])
        
        self.M = cv2.getPerspectiveTransform(self.src, self.dst)
        self.M_inv = cv2.getPerspectiveTransform(self.dst, self.src)

    def forward(self, img, flags=cv2.INTER_LINEAR):
        """ Take a front view image and transform to top view

        Parameters:
            img (np.array): A front view image
            flags : flag to use in cv2.warpPerspective()

        Returns:
            Image (np.array): Top view image
        """
        if self.M is None:
            self._calculate_transform(img.shape)
        img_size = (img.shape[1], img.shape[0])
        return cv2.warpPerspective(img, self.M, img_size, flags=flags)

    def backward(self, img, flags=cv2.INTER_LINEAR):
        """ Take a top view image and transform it to front view

        Parameters:
            img (np.array): A top view image
            flags (int): flag to use in cv2.warpPerspective()

        Returns:
            Image (np.array): Front view image
        """
        if self.M_inv is None:
            self._calculate_transform(img.shape)
        img_size = (img.shape[1], img.shape[0])
        return cv2.warpPerspective(img, self.M_inv, img_size, flags=flags) 