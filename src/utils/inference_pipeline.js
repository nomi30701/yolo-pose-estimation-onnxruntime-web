import cvReadyPromise from "@techstark/opencv-js";
import { preProcess_img, applyNMS, Colors } from "./img_preprocess";

let cv;

// init opencvjs
(async () => {
  cv = await cvReadyPromise;
})();

/**
 * Inference pipeline for YOLO model.
 * @param {HTMLImageElement|HTMLCanvasElement|OffscreenCanvas} imageSource - Input image source
 * @param {ort.InferenceSession} session - YOLO model ort session.
 * @param {[Number, Number]} overlay_size - Overlay width and height. [width, height]
 * @param {object} model_config - Model configuration object.
 * @returns {[object, string]} Tuple containing:
 *   - First element: object with inference results:
 *     - bbox_results: Array<Object> - Filtered detection results after NMS, each containing:
 *       - bbox: [x, y, width, height] in original image coordinates
 *       - class_idx: Predicted class index
 *       - score: Confidence score (0-1)
 *       - keypoints:  For pose tasks: [{x, y, score}] for each keypoint
 *     - mask_imgData?: For segmentation tasks: RGBA overlay image with colored masks
 *   - Second element: Inference time in milliseconds (formatted to 2 decimal places)
 *
 */
export async function inference_pipeline(
  imageSource,
  session,
  overlay_size,
  model_config
) {
  try {
    // Read DOM to cv.Mat
    const src_mat = cv.imread(imageSource);

    // Pre-process img, inference
    const [input_tensor, xRatio, yRatio] = preProcess_img(
      src_mat,
      overlay_size,
      model_config.imgsz_type
    );
    src_mat.delete();

    const start = performance.now();
    const { output0 } = await session.run({
      images: input_tensor,
    });
    const end = performance.now();
    input_tensor.dispose();

    // Post process
    const results = postProcess_pose(
      output0,
      model_config.score_threshold,
      xRatio,
      yRatio
    );

    output0.dispose();

    // Apply NMS
    const selected_indices = applyNMS(
      results,
      results.map((r) => r.score),
      model_config.iou_threshold
    );
    const filtered_results = selected_indices.map((i) => results[i]);

    return [filtered_results, (end - start).toFixed(2)];
  } catch (error) {
    console.error("Inference error:", error);
    return [[], "0.00"];
  }
}

/**
 *
 * @param {ort.Tensor} raw_tensor - Yolo model output0
 * @param {number} score_threshold - Score threshold
 * @param {number} xRatio - xRatio
 * @param {number} yRatio - yRatio
 * @returns {Array<Object>} Array of pose estimation results.
 * - [{bbox: [x, y, w, h], score, keypoints: [{x, y, score}]}, ...]
 */
function postProcess_pose(raw_tensor, score_threshold = 0.45, xRatio, yRatio) {
  // post process
  const NUM_PREDICTIONS = raw_tensor.dims[2];
  const NUM_BBOX_ATTRS = 5;
  const NUM_KEYPOINTS = 17;
  const KEYPOINT_DIMS = 3;

  const predictions = raw_tensor.data;
  const bbox_data = predictions.subarray(0, NUM_PREDICTIONS * NUM_BBOX_ATTRS);
  const keypoints_data = predictions.subarray(NUM_PREDICTIONS * NUM_BBOX_ATTRS);

  const results = new Array();
  let resultCount = 0;

  for (let i = 0; i < NUM_PREDICTIONS; i++) {
    const score = bbox_data[i + NUM_PREDICTIONS * 4];
    if (score <= score_threshold) continue;

    const w = bbox_data[i + NUM_PREDICTIONS * 2] * xRatio;
    const h = bbox_data[i + NUM_PREDICTIONS * 3] * yRatio;
    const tlx = bbox_data[i] * xRatio - 0.5 * w;
    const tly = bbox_data[i + NUM_PREDICTIONS] * yRatio - 0.5 * h;

    const keypoints = new Array(NUM_KEYPOINTS);
    for (let kp = 0; kp < NUM_KEYPOINTS; kp++) {
      const base_idx = kp * KEYPOINT_DIMS * NUM_PREDICTIONS + i;
      keypoints[kp] = {
        x: keypoints_data[base_idx] * xRatio,
        y: keypoints_data[base_idx + NUM_PREDICTIONS] * yRatio,
        score: keypoints_data[base_idx + NUM_PREDICTIONS * 2],
      };
    }

    results[resultCount++] = {
      bbox: [tlx, tly, w, h],
      score,
      keypoints,
    };
  }
  return results;
}
