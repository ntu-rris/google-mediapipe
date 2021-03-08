// Our input frames will come from here.
const videoElement = document.getElementsByClassName("input_video")[0];
const canvasElement = document.getElementsByClassName("output_canvas")[0];
const controlsElement = document.getElementsByClassName("control-panel")[0];
const canvasCtx = canvasElement.getContext("2d");

// We'll add this to our control panel later, but we'll save it here so we can
// call tick() each time the graph runs.
const fpsControl = new FPS();

// Optimization: Turn off animated spinner after its hiding animation is done.
const spinner = document.querySelector(".loading");
spinner.ontransitionend = () => {
  spinner.style.display = "none";
};

function zColor(data) {
  const z = clamp(data.from.z + 0.5, 0, 1);
  return `rgba(0, ${255 * z}, ${255 * (1 - z)}, 1)`;
}

function onResults(results) {
  // Hide the spinner.
  document.body.classList.add("loaded");

  // Update the frame rate.
  fpsControl.tick();

  // Draw the overlays.
  canvasCtx.save();
  canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
  canvasCtx.drawImage(
    results.image,
    0,
    0,
    canvasElement.width,
    canvasElement.height
  );
  drawConnectors(canvasCtx, results.poseLandmarks, POSE_CONNECTIONS, {
    visibilityMin: 0.65,
    color: (data) => {
      const x0 = canvasElement.width * data.from.x;
      const y0 = canvasElement.height * data.from.y;
      const x1 = canvasElement.width * data.to.x;
      const y1 = canvasElement.height * data.to.y;

      const z0 = clamp(data.from.z + 0.5, 0, 1);
      const z1 = clamp(data.to.z + 0.5, 0, 1);

      const gradient = canvasCtx.createLinearGradient(x0, y0, x1, y1);
      gradient.addColorStop(0, `rgba(0, ${255 * z0}, ${255 * (1 - z0)}, 1)`);
      gradient.addColorStop(1.0, `rgba(0, ${255 * z1}, ${255 * (1 - z1)}, 1)`);
      return gradient;
    }
  });
  drawLandmarks(
    canvasCtx,
    Object.values(POSE_LANDMARKS_LEFT).map(
      (index) => results.poseLandmarks[index]
    ),
    { visibilityMin: 0.65, color: zColor, fillColor: "#FF0000" }
  );
  drawLandmarks(
    canvasCtx,
    Object.values(POSE_LANDMARKS_RIGHT).map(
      (index) => results.poseLandmarks[index]
    ),
    { visibilityMin: 0.65, color: zColor, fillColor: "#00FF00" }
  );
  drawLandmarks(
    canvasCtx,
    Object.values(POSE_LANDMARKS_NEUTRAL).map(
      (index) => results.poseLandmarks[index]
    ),
    { visibilityMin: 0.65, color: zColor, fillColor: "#AAAAAA" }
  );
  canvasCtx.restore();

  // Compute elbow joint angle
  leftElbowAngle = getFlexionAng(results, 11,13,15);
  rightElbowAngle = getFlexionAng(results, 12,14,16);

  // Display elbow joint angle
  document.getElementById('left-elbow-angle').innerHTML = 'Left elbow angle: ' + leftElbowAngle + ' deg';
  document.getElementById('right-elbow-angle').innerHTML = 'Right elbow angle: ' + rightElbowAngle + ' deg';
}

const pose = new Pose({
  locateFile: (file) => {
    return `https://cdn.jsdelivr.net/npm/@mediapipe/pose@0.2/${file}`;
  }
});
pose.onResults(onResults);

/**
 * Instantiate a camera. We'll feed each frame we receive into the solution.
 */
const camera = new Camera(videoElement, {
  onFrame: async () => {
    await pose.send({ image: videoElement });
  },
  width: 1280,
  height: 720
});
camera.start();

// Present a control panel through which the user can manipulate the solution
// options.
new ControlPanel(controlsElement, {
  selfieMode: true,
  upperBodyOnly: true,
  smoothLandmarks: true,
  minDetectionConfidence: 0.5,
  minTrackingConfidence: 0.5,
  pauseCamera: false,
})
  .add([
    new StaticText({ title: "Elbow joint angle" }),
    fpsControl,
    new Toggle({ title: "Selfie Mode", field: "selfieMode" }),
    new Toggle({ title: "Upper-body Only", field: "upperBodyOnly" }),
    new Toggle({ title: "Smooth Landmarks", field: "smoothLandmarks" }),
    new Toggle({ title: "Pause Camera", field: "pauseCamera" }),
    // new Slider({
    //   title: "Min Detection Confidence",
    //   field: "minDetectionConfidence",
    //   range: [0, 1],
    //   step: 0.01
    // }),
    // new Slider({
    //   title: "Min Tracking Confidence",
    //   field: "minTrackingConfidence",
    //   range: [0, 1],
    //   step: 0.01
    // })
  ])
  .on((options) => {
    videoElement.classList.toggle("selfie", options.selfieMode);
    pose.setOptions(options);
    // Add option to pause camera
    options.pauseCamera ? camera.video.pause() : camera.start();
  });

// Added by GM
function getFlexionAng(results, j0, j1, j2) {
  // j0, j1, j2 are joint indexes of poseLandmarks
  // 1st: rescale landmarks from relative coor to absolute 3D coor
  var A = rescale(results.poseLandmarks[j0]);
  var B = rescale(results.poseLandmarks[j1]);
  var C = rescale(results.poseLandmarks[j2]);
  // 2nd: Find the acute angle at joint j1
  return getAngBtw3Pts(A, B, C).toFixed(0);
}

function rescale(lm) {
  // Convert landmarks from relative coor to absolute 3D coor
  // Note: Assume image width:1280, height:720
  var A = {x:0, y:0, z:0};
  A.x = lm.x * 1280 - 640;
  A.y = lm.y * 720 - 360;
  A.z = lm.z * 1280 * 0.25; // Note: Seems like need to further scale down z by 0.25 else will get elongated forearm and feet
  return A;
}

function getAngBtw3Pts(A, B, C) {
  // Note: Points A, B and C are 3D points with x,y,z
  // 1st: Find vector AB (a,b,c) and BC (d,e,f)
  // 2nd: Find acute angle between vector AB and BC using cosine rule
  
  // Find vector AB = OB - OA
  var a = B.x - A.x;
  var b = B.y - A.y;
  var c = B.z - A.z;
  // Find vector BC = OC - OB
  var d = C.x - B.x;
  var e = C.y - B.y;
  var f = C.z - B.z;
  
  // BA.BC -> dot product
  var num = a*d + b*e + c*f;
  // |BA|*|BC| -> multiply norm of BA and BC
  var den = Math.sqrt(a*a + b*b + c*c) * Math.sqrt(d*d + e*e + f*f);
  // ang = acos(BA.BC / |BA|*|BC|)
  var ang = Math.acos(num/den);
  
  // Convert radian to degree  
  return ang * 180 / Math.PI;
}