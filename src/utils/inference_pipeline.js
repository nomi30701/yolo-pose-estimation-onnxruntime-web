import * as ort from "onnxruntime-web/webgpu";

/**
 * @typedef {Object} config
 * @property {[Number]} input_shape  - input shape of the model.
 * @property {Number} iou_threshold - Intersection over union threshold.
 * @property {Number} score_threshold - Score threshold.
 */
/**
 * Yolov11 inference pipeline.
 * @param {(HTMLImageElement|HTMLCanvasElement)} input_el - Input <img> or <canvas> element for detect.
 * @param {ort.InferenceSession} session - Yolo model.
 * @param {sessionsConfig} config - Configuration for the model.
 *
 * @returns {[Array[Object], Number]} - Array of predictions and inference time.
 */
export const inference_pipeline = async (input_el, session, config) => {
  const src_mat = cv.imread(input_el);

  // pre process input image
  // const [src_mat_preProcessed, xRatio, yRatio] = await preProcess(
  //   src_mat,
  //   sessionsConfig.input_shape[2],
  //   sessionsConfig.input_shape[3]
  // );

  const [src_mat_preProcessed, div_width, div_height] =
    preProcess_dynamic(src_mat);
  const xRatio = src_mat.cols / div_width;
  const yRatio = src_mat.rows / div_height;

  src_mat.delete();

  const input_tensor = new ort.Tensor("float32", src_mat_preProcessed.data32F, [
    1,
    3,
    div_height,
    div_width,
  ]);
  src_mat_preProcessed.delete();

  // inference
  const start = performance.now();
  const { output0 } = await session.run({
    images: input_tensor,
  });
  const end = performance.now();
  input_tensor.dispose();

  // post process
  const num_predictions = output0.dims[2];
  const NUM_BBOX_ATTRS = 5;
  const NUM_KEYPOINTS = 17;
  const KEYPOINT_DIMS = 3;

  const predictions = output0.data;
  const bbox_data = predictions.subarray(0, num_predictions * NUM_BBOX_ATTRS);
  const keypoints_data = predictions.subarray(num_predictions * NUM_BBOX_ATTRS);

  const results = [];
  for (let i = 0; i < num_predictions; i++) {
    const score = bbox_data[i + num_predictions * 4];
    if (score <= config.score_threshold) continue;

    const x =
      (bbox_data[i] - 0.5 * bbox_data[i + num_predictions * 2]) * xRatio;
    const y =
      (bbox_data[i + num_predictions] -
        0.5 * bbox_data[i + num_predictions * 3]) *
      yRatio;
    const w = bbox_data[i + num_predictions * 2] * xRatio;
    const h = bbox_data[i + num_predictions * 3] * yRatio;

    const keypoints = new Array(NUM_KEYPOINTS);
    for (let kp = 0; kp < NUM_KEYPOINTS; kp++) {
      const base_idx = kp * KEYPOINT_DIMS * num_predictions + i;
      keypoints[kp] = {
        x: keypoints_data[base_idx] * xRatio,
        y: keypoints_data[base_idx + num_predictions] * yRatio,
        score: keypoints_data[base_idx + num_predictions * 2],
      };
    }

    results.push({
      bbox: [x, y, w, h],
      score,
      keypoints,
    });
  }

  const selected_indices = applyNMS(
    results,
    results.map((r) => r.score),
    config.iou_threshold
  );
  const filtered_results = selected_indices.map((i) => results[i]);

  return [filtered_results, (end - start).toFixed(2)];
};

function calculateIOU(box1, box2) {
  const [x1, y1, w1, h1] = box1;
  const [x2, y2, w2, h2] = box2;

  const box1_x2 = x1 + w1;
  const box1_y2 = y1 + h1;
  const box2_x2 = x2 + w2;
  const box2_y2 = y2 + h2;

  const intersect_x1 = Math.max(x1, x2);
  const intersect_y1 = Math.max(y1, y2);
  const intersect_x2 = Math.min(box1_x2, box2_x2);
  const intersect_y2 = Math.min(box1_y2, box2_y2);

  if (intersect_x2 <= intersect_x1 || intersect_y2 <= intersect_y1) {
    return 0.0;
  }

  const intersection =
    (intersect_x2 - intersect_x1) * (intersect_y2 - intersect_y1);
  const box1_area = w1 * h1;
  const box2_area = w2 * h2;

  return intersection / (box1_area + box2_area - intersection);
}

function applyNMS(boxes, scores, iou_threshold = 0.35) {
  const picked = [];
  const indexes = Array.from(Array(scores.length).keys());

  indexes.sort((a, b) => scores[b] - scores[a]);

  while (indexes.length > 0) {
    const current = indexes[0];
    picked.push(current);

    const rest = indexes.slice(1);
    indexes.length = 0;

    for (const idx of rest) {
      const iou = calculateIOU(boxes[current].bbox, boxes[idx].bbox);
      if (iou <= iou_threshold) {
        indexes.push(idx);
      }
    }
  }

  return picked;
}

/**
 * Pre process input image.
 *
 * Resize and normalize image.
 *
 *
 * @param {cv.Mat} mat - Pre process yolo model input image.
 * @param {Number} input_width - Yolo model input width.
 * @param {Number} input_height - Yolo model input height.
 * @returns {cv.Mat} Processed input mat.
 */
const preProcess = (mat, input_width, input_height) => {
  cv.cvtColor(mat, mat, cv.COLOR_RGBA2RGB);

  // Resize to dimensions divisible by 32
  const [div_width, div_height] = divStride(32, mat.cols, mat.rows);
  cv.resize(mat, mat, new cv.Size(div_width, div_height));

  // Padding to square
  const max_dim = Math.max(div_width, div_height);
  const right_pad = max_dim - div_width;
  const bottom_pad = max_dim - div_height;
  cv.copyMakeBorder(
    mat,
    mat,
    0,
    bottom_pad,
    0,
    right_pad,
    cv.BORDER_CONSTANT,
    new cv.Scalar(0, 0, 0)
  );

  // Calculate ratios
  const xRatio = mat.cols / input_width;
  const yRatio = mat.rows / input_height;

  // Resize to input dimensions and normalize to [0, 1]
  const preProcessed = cv.blobFromImage(
    mat,
    1 / 255.0,
    new cv.Size(input_width, input_height),
    new cv.Scalar(0, 0, 0),
    false,
    false
  );

  return [preProcessed, xRatio, yRatio];
};

/**
 * Pre process input image.
 *
 * Normalize image.
 *
 * @param {cv.Mat} mat - Pre process yolo model input image.
 * @param {Number} input_width - Yolo model input width.
 * @param {Number} input_height - Yolo model input height.
 * @returns {cv.Mat} Processed input mat.
 */
const preProcess_dynamic = (mat) => {
  cv.cvtColor(mat, mat, cv.COLOR_RGBA2RGB);

  // resize image to divisible by 32
  const [div_width, div_height] = divStride(32, mat.cols, mat.rows);
  // resize, normalize to [0, 1]
  const preProcessed = cv.blobFromImage(
    mat,
    1 / 255.0,
    new cv.Size(div_width, div_height),
    new cv.Scalar(0, 0, 0),
    false,
    false
  );
  return [preProcessed, div_width, div_height];
};

/**
 * Return height and width are divisible by stride.
 * @param {Number} stride - Stride value.
 * @param {Number} width - Image width.
 * @param {Number} height - Image height.
 * @returns {[Number]}[width, height] divisible by stride.
 **/
const divStride = (stride, width, height) => {
  width =
    width % stride >= stride / 2
      ? (Math.floor(width / stride) + 1) * stride
      : Math.floor(width / stride) * stride;

  height =
    height % stride >= stride / 2
      ? (Math.floor(height / stride) + 1) * stride
      : Math.floor(height / stride) * stride;

  return [width, height];
};
