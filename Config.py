from mrcnn.config import Config
class DefectConfig(Config):
    # Give the configuration a recognizable name
    NAME = "fabric_defect_segmentation"

    NUM_CLASSES = 1 + 1

    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


config = DefectConfig()
# config.display()