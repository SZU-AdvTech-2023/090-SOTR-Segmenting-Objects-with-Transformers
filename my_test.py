import cv2
from detectron2.utils.logger import setup_logger
from demo.predictor import VisualizationDemo
from adet.config import get_cfg
from torchvision import models

def setup_cfg(config_file, model_weights):
    # Load config from file
    cfg = get_cfg()
    cfg.merge_from_file(config_file)
    cfg.MODEL.WEIGHTS = model_weights
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.55  # Set a confidence threshold
    cfg.freeze()
    return cfg


def process_image(cfg, image_path):
    # Create a demo object
    demo = VisualizationDemo(cfg)

    # Load image and run model inference
    img = cv2.imread(image_path)
    predictions, visualized_output = demo.run_on_image(img)

    return visualized_output.get_image()[:, :, ::-1]


if __name__ == "__main__":

    # 加载预训练的 ResNeXt101 模型
    #model = models.resnext101_32x8d(pretrained=True)
    
    
    
    config_files = [
        "/data3/fengjunyuan/PycharmProjects/SOTR/configs/SOTR/R50.yaml",
        "/data3/fengjunyuan/PycharmProjects/SOTR/configs/SOTR/R101.yaml",
        "/data3/fengjunyuan/PycharmProjects/SOTR/configs/SOTR/R_101_DCN.yaml"
    ]
    model_weights = [
        "/data3/fengjunyuan/PycharmProjects/SOTR/work_dir/SOTR_R50/SOTR_R50.pth",
        "/data3/fengjunyuan/PycharmProjects/SOTR/work_dir/SOTR_R101/SOTR_R101.pth",
        "/data3/fengjunyuan/PycharmProjects/SOTR/work_dir/SOTR_R101_DCN/SOTR_R101_DCN.pth"
    ]

    image_path ="/data3/fengjunyuan/PycharmProjects/SOTR/datasets/coco/val2017/000000008021.jpg"

    # Setup logger
    logger = setup_logger()

    # Process and display images for each model
    for config_file, weight in zip(config_files, model_weights):
        cfg = setup_cfg(config_file, weight)
        visualized_image = process_image(cfg, image_path)
        cv2.imshow("Model Output", visualized_image)
        cv2.waitKey(0)

    cv2.destroyAllWindows()
