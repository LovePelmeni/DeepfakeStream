import abc 
from abc import abstractmethod, abstractproperty

class BaseDriftDetector(abc.ABC):

    @abstractmethod
    def detect(self,  **kwargs):
        """
        Function detects data drift,
        by comparing entities, provided under **kwargs section.
        It can be distributions, real-value metrics or whatever, 
        that can be matched against baseline.
        """

    @abstractproperty
    def description(self):
        """
        Returns / prints 
        description document, which 
        explains key data properties the detector
        class is responsible for checking, including test strategies involved.
        
        Example:
            ConceptDriftDetector - drift detector, responsible
            for dealing with feature distribution changing over time.

            Keypoints to detect: [feature1, feature2, featuren, ...].
            feature1 - (description) + (testing method used)
            feature2 - (description) + (testing method used)
            featuren - (description for featureN) + (testing method, used for featureN)

            Assumptions:
                (
                    define the area, which is out of scope of the detector
                    and what limitations / constraints it has.
                )
        """

