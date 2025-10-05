import { Colors } from "./img_preprocess.js";

/**
 * Draw bounding boxes in overlay canvas based on task type.
 * @param {Array[Object]} predictions - Detection/pose results
 * @param {HTMLCanvasElement} overlay_ctx - Show boxes in overlay canvas element
 */
export async function render_overlay(predictions, overlay_ctx) {
  // Calculate diagonal length of the canvas
  const diagonalLength = Math.sqrt(
    Math.pow(overlay_ctx.canvas.width, 2) +
      Math.pow(overlay_ctx.canvas.height, 2)
  );
  const lineWidth = diagonalLength / 250;

  if (!predictions || predictions.length === 0) return;

  // Draw predictions
  draw_pose_estimation(overlay_ctx, predictions, lineWidth);
}

/**
 * Draw pose estimation results
 */
function draw_pose_estimation(ctx, predictions, lineWidth) {
  if (!predictions || predictions.length === 0) return;

  // draw all bounding boxes
  ctx.lineWidth = lineWidth;
  ctx.strokeStyle = "green";
  predictions.forEach((predict) => {
    const [x1, y1, width, height] = predict.bbox;
    ctx.strokeRect(x1, y1, width, height);
  });

  // 2. draw all scores
  ctx.fillStyle = "green";
  ctx.font = fontCache.font;
  predictions.forEach((predict) => {
    const [x1, y1] = predict.bbox;
    const text = `score ${predict.score.toFixed(2)}`;
    drawTextWithBackground(ctx, text, x1, y1);
  });

  // 3. draw all skeletons
  ctx.strokeStyle = "rgb(255, 165, 0)";
  ctx.lineWidth = 2;
  ctx.beginPath();

  const SKELETON = [
    [15, 13],
    [13, 11],
    [16, 14],
    [14, 12],
    [11, 12],
    [5, 11],
    [6, 12],
    [5, 6],
    [5, 7],
    [6, 8],
    [7, 9],
    [8, 10],
    [1, 2],
    [0, 1],
    [0, 2],
    [1, 3],
    [2, 4],
    [3, 5],
    [4, 6],
  ];

  // connect all keypoints
  predictions.forEach((predict) => {
    if (!predict.keypoints) return;

    SKELETON.forEach(([i, j]) => {
      const kp1 = predict.keypoints[i];
      const kp2 = predict.keypoints[j];

      if (kp1 && kp2 && kp1.score > 0.5 && kp2.score > 0.5) {
        ctx.moveTo(kp1.x, kp1.y);
        ctx.lineTo(kp2.x, kp2.y);
      }
    });
  });
  ctx.stroke();

  // draw all keypoints
  ctx.fillStyle = "red";
  predictions.forEach((predict) => {
    if (!predict.keypoints) return;

    predict.keypoints.forEach((keypoint) => {
      const { x, y, score } = keypoint;
      if (score < 0.5) return;

      ctx.beginPath();
      ctx.arc(x, y, 3, 0, 2 * Math.PI);
      ctx.fill();
    });
  });
}

const fontCache = {
  font: "16px Arial",
  measurements: {},
};

function getMeasuredTextWidth(text, ctx) {
  if (!fontCache.measurements[text]) {
    fontCache.measurements[text] = ctx.measureText(text).width;
  }
  return fontCache.measurements[text];
}

/**
 * Helper function to draw text with background
 */
function drawTextWithBackground(ctx, text, x, y) {
  ctx.font = fontCache.font;
  const textWidth = getMeasuredTextWidth(text, ctx);
  const textHeight = 16;

  // Calculate the Y position for the text
  let textY = y - 5;
  let rectY = y - textHeight - 4;

  // Check if the text will be outside the canvas
  if (rectY < 0) {
    // Adjust the Y position to be inside the canvas
    textY = y + textHeight + 5;
    rectY = y + 1;
  }

  const currentFillStyle = ctx.fillStyle;
  ctx.fillRect(x - 1, rectY, textWidth + 4, textHeight + 4);
  ctx.fillStyle = "white";
  ctx.fillText(text, x, textY);
  ctx.fillStyle = currentFillStyle;
}
