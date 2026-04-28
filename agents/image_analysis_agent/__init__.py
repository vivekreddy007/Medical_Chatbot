from .image_classifier import ImageClassifier
from .brain_tumor_agent.brain_tumor_inference import BrainTumorSegmentation
from .chest_xray_agent.covid_chest_xray_inference import ChestXRayClassification
from .skin_lesion_agent.skin_lesion_inference import SkinLesionSegmentation


class ImageAnalysisAgent:
    """Processes image uploads: classifies type and runs the appropriate CV model."""

    def __init__(self, config):
        self.image_classifier = ImageClassifier(vision_model=config.medical_cv.llm)
        self.brain_tumor_agent = BrainTumorSegmentation(model_path=config.medical_cv.brain_tumor_model_path)
        self.chest_xray_agent = ChestXRayClassification(model_path=config.medical_cv.chest_xray_model_path)
        self.skin_lesion_agent = SkinLesionSegmentation(model_path=config.medical_cv.skin_lesion_model_path)
        self.brain_tumor_output_path = config.medical_cv.brain_tumor_segmentation_output_path
        self.skin_lesion_output_path = config.medical_cv.skin_lesion_segmentation_output_path

    def analyze_image(self, image_path: str) -> dict:
        return self.image_classifier.classify_image(image_path)

    def segment_brain_tumor(self, image_path: str) -> bool:
        return self.brain_tumor_agent.predict(image_path, self.brain_tumor_output_path)

    def classify_chest_xray(self, image_path: str) -> str:
        return self.chest_xray_agent.predict(image_path)

    def segment_skin_lesion(self, image_path: str) -> bool:
        return self.skin_lesion_agent.predict(image_path, self.skin_lesion_output_path)
