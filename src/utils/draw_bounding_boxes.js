/**
 * Draw bounding boxes in overlay canvas.
 * @param {Array[Object]} predictions - Bounding boxes, score and keypoint.
 * @param {HTMLCanvasElement} overlay_el - Show boxes in overlay canvas element.
 */

export async function draw_bounding_boxes(predictions, overlay_el) {
  const ctx = overlay_el.getContext("2d");

  // Clear the canvas
  ctx.clearRect(0, 0, overlay_el.width, overlay_el.height);

  // Calculate diagonal length of the canvas
  const diagonalLength = Math.sqrt(
    Math.pow(overlay_el.width, 2) + Math.pow(overlay_el.height, 2)
  );
  const lineWidth = diagonalLength / 250;

  // Draw boxes and keypoint
  predictions.forEach((predict) => {
    const [x1, y1, width, height] = predict.bbox;

    // Draw bbox
    ctx.lineWidth = lineWidth;
    ctx.strokeStyle = "green";
    ctx.strokeRect(x1, y1, width, height);

    // Draw text and background
    ctx.fillStyle = "green";
    ctx.font = "16px Arial";
    const text = `score ${predict.score.toFixed(2)}`;
    const textWidth = ctx.measureText(text).width;
    const textHeight = parseInt(ctx.font, 10);

    // Calculate the Y position for the text
    let textY = y1 - 5;
    let rectY = y1 - textHeight - 4;

    // if the text outside the canvas
    if (rectY < 0) {
      // Adjust the Y position to be inside the canvas
      textY = y1 + textHeight + 5;
      rectY = y1 + 1;
    }

    ctx.fillRect(x1 - 1, rectY, textWidth + 4, textHeight + 4);
    ctx.fillStyle = "white";
    ctx.fillText(text, x1, textY);

    // Draw keypoints and connections
    const keypoints = predict.keypoints;

    // draw connections
    ctx.strokeStyle = "rgb(255, 165, 0)";
    ctx.lineWidth = 2;
    SKELETON.forEach(([i, j]) => {
      const kp1 = keypoints[i];
      const kp2 = keypoints[j];

      // if score is low, ignore
      if (kp1.score > 0.5 && kp2.score > 0.5) {
        ctx.beginPath();
        ctx.moveTo(kp1.x, kp1.y);
        ctx.lineTo(kp2.x, kp2.y);
        ctx.stroke();
      }
    });

    // draw keypoints
    keypoints.forEach((keypoint) => {
      const { x, y, score } = keypoint;
      if (score < 0.5) return;
      ctx.beginPath();
      ctx.arc(x, y, 3, 0, 2 * Math.PI);
      ctx.fillStyle = "red";
      ctx.fill();
    });
  });
}

const SKELETON = [
  [15, 13],
  [13, 11],
  [16, 14],
  [14, 12], // leg
  [11, 12], // butts
  [5, 11],
  [6, 12], // body
  [5, 6], // shoulder
  [5, 7],
  [6, 8],
  [7, 9],
  [8, 10], // arms
  [1, 2],
  [0, 1],
  [0, 2],
  [1, 3],
  [2, 4], // face
  [3, 5],
  [4, 6], // ear to shoulder
];
