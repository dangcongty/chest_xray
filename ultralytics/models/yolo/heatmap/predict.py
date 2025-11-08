# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from ultralytics.engine.results import Results
from ultralytics.models.yolo.detect.predict import DetectionPredictor
from ultralytics.utils import DEFAULT_CFG, ops


class HeatmapPredictor(DetectionPredictor):
    """A class extending the DetectionPredictor class for prediction based on a Heatmap model.

    This class specializes in processing Heatmap model outputs, handling both bounding boxes and masks in the
    prediction results.

    Attributes:
        args (dict): Configuration arguments for the predictor.
        model (torch.nn.Module): The loaded YOLO Heatmap model.
        batch (list): Current batch of images being processed.

    Methods:
        postprocess: Apply non-max suppression and process Heatmap detections.
        construct_results: Construct a list of result objects from predictions.
        construct_result: Construct a single result object from a prediction.

    Examples:
        >>> from ultralytics.utils import ASSETS
        >>> from ultralytics.models.yolo.segment import HeatmapPredictor
        >>> args = dict(model="yolo11n-seg.pt", source=ASSETS)
        >>> predictor = HeatmapPredictor(overrides=args)
        >>> predictor.predict_cli()
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """Initialize the HeatmapPredictor with configuration, overrides, and callbacks.

        This class specializes in processing Heatmap model outputs, handling both bounding boxes and masks in the
        prediction results.

        Args:
            cfg (dict): Configuration for the predictor.
            overrides (dict, optional): Configuration overrides that take precedence over cfg.
            _callbacks (list, optional): List of callback functions to be invoked during prediction.
        """
        super().__init__(cfg, overrides, _callbacks)
        self.args.task = "heatmap"

    def postprocess(self, preds, img, orig_imgs):
        return super().postprocess(preds[0], img, orig_imgs)

    def construct_results(self, preds, img, orig_imgs, protos):
        return [
            self.construct_result(pred, img, orig_img, img_path, proto)
            for pred, orig_img, img_path, proto in zip(preds, orig_imgs, self.batch[0], protos)
        ]

    def construct_result(self, pred, img, orig_img, img_path, proto):
        """Construct a single result object from the prediction.

        Args:
            pred (torch.Tensor): The predicted bounding boxes, scores, and masks.
            img (torch.Tensor): The image after preprocessing.
            orig_img (np.ndarray): The original image before preprocessing.
            img_path (str): The path to the original image.
            proto (torch.Tensor): The prototype masks.

        Returns:
            (Results): Result object containing the original image, image path, class names, bounding boxes, and masks.
        """
        if pred.shape[0] == 0:  # save empty boxes
            masks = None
        elif self.args.retina_masks:
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
            masks = ops.process_mask_native(proto, pred[:, 6:], pred[:, :4], orig_img.shape[:2])  # HWC
        else:
            masks = ops.process_mask(proto, pred[:, 6:], pred[:, :4], img.shape[2:], upsample=True)  # HWC
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
        if masks is not None:
            keep = masks.amax((-2, -1)) > 0  # only keep predictions with masks
            if not all(keep):  # most predictions have masks
                pred, masks = pred[keep], masks[keep]  # indexing is slow
        return Results(orig_img, path=img_path, names=self.model.names, boxes=pred[:, :6], masks=masks)
