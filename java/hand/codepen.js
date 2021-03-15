// Our input frames will come from here.
const videoElement = document.getElementsByClassName('input_video')[0];
const canvasElement = document.getElementsByClassName('output_canvas')[0];
const controlsElement = document.getElementsByClassName('control-panel')[0];
const canvasCtx = canvasElement.getContext('2d');

// We'll add this to our control panel later, but we'll save it here so we can
// call tick() each time the graph runs.
const fpsControl = new FPS();

// Optimization: Turn off animated spinner after its hiding animation is done.
const spinner = document.querySelector('.loading');
spinner.ontransitionend = () => {
  spinner.style.display = 'none';
};

// Exponential filter to filter out high frequency noise
var alpha = 0.5;
var thumb1 = 0;
var thumb2 = 0;
var thumb3 = 0;
var index1 = 0;
var index2 = 0;
var index3 = 0;
var middle1 = 0;
var middle2 = 0;
var middle3 = 0;
var ring1 = 0;
var ring2 = 0;
var ring3 = 0;
var little1 = 0;
var little2 = 0;
var little3 = 0;

// For progress display
var progress1 = 0;
var progress2 = 0;
var progress3 = 0;
var progress4 = 0;
var progress5 = 0;

// For playing sound
var enableSound = false;
var soundThres = 20;
var sound1 = new Audio('https://assets.codepen.io/5628206/beat_1.mp3');
var sound2 = new Audio('https://assets.codepen.io/5628206/beat_2.mp3');
var sound3 = new Audio('https://assets.codepen.io/5628206/beat_3.mp3');
var sound4 = new Audio('https://assets.codepen.io/5628206/beat_4.mp3');
var sound5 = new Audio('https://assets.codepen.io/5628206/beat_5.mp3');
var play1 = false;
var play2 = false;
var play3 = false;
var play4 = false;
var play5 = false;

function onResults(results) {
  // Hide the spinner.
  document.body.classList.add('loaded');

  // Update the frame rate.
  fpsControl.tick();

  // Draw the overlays.
  canvasCtx.save();
  canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
  canvasCtx.drawImage(
      results.image, 0, 0, canvasElement.width, canvasElement.height);
  if (results.multiHandLandmarks && results.multiHandedness) {
    for (let index = 0; index < results.multiHandLandmarks.length; index++) {
      const classification = results.multiHandedness[index];
      const isRightHand = classification.label === 'Right';
      const landmarks = results.multiHandLandmarks[index];
      drawConnectors(
          canvasCtx, landmarks, HAND_CONNECTIONS,
          {color: isRightHand ? '#00FF00' : '#FF0000'}),
      drawLandmarks(canvasCtx, landmarks, {
        color: isRightHand ? '#00FF00' : '#FF0000',
        fillColor: isRightHand ? '#FF0000' : '#00FF00',
        radius: (x) => {
          return lerp(x.from.z, -0.15, .1, 10, 1);
        }
      });
    }
    
    // Compute hand joint flexion angles
    // thumb1 = alpha*getFlexionAng(results, 0,1,2) + (1-alpha)*thumb1; // CMC hide this value as it is not accurate
    thumb2 = alpha*getFlexionAng(results, 1,2,3) + (1-alpha)*thumb2; // MCP
    thumb3 = alpha*getFlexionAng(results, 2,3,4) + (1-alpha)*thumb3; // IP

    index1 = alpha*getFlexionAng(results, 0,5,6) + (1-alpha)*index1; // MCP
    index2 = alpha*getFlexionAng(results, 5,6,7) + (1-alpha)*index2; // PIP
    index3 = alpha*getFlexionAng(results, 6,7,8) + (1-alpha)*index3; // DIP

    middle1 = alpha*getFlexionAng(results,  0, 9,10) + (1-alpha)*middle1; // MCP
    middle2 = alpha*getFlexionAng(results,  9,10,11) + (1-alpha)*middle2; // PIP
    middle3 = alpha*getFlexionAng(results, 10,11,12) + (1-alpha)*middle3; // DIP

    ring1 = alpha*getFlexionAng(results,  0,13,14) + (1-alpha)*ring1; // MCP
    ring2 = alpha*getFlexionAng(results, 13,14,15) + (1-alpha)*ring2; // PIP
    ring3 = alpha*getFlexionAng(results, 14,15,16) + (1-alpha)*ring3; // DIP

    little1 = alpha*getFlexionAng(results,  0,17,18) + (1-alpha)*little1; // MCP
    little2 = alpha*getFlexionAng(results, 17,18,19) + (1-alpha)*little2; // PIP
    little3 = alpha*getFlexionAng(results, 18,19,20) + (1-alpha)*little3; // DIP

    // Display hand joint angles
    document.getElementById('Thumb' ).innerHTML = 
      'T. MCP: ' + 
      padLeadingZeros(thumb2.toFixed(0), 3)  + 
      '  IP: ' + 
      padLeadingZeros(thumb3.toFixed(0), 3)  + ' deg';
    document.getElementById('Index' ).innerHTML = 
      'I. MCP: ' + 
      padLeadingZeros(index1.toFixed(0), 3)  + 
      ' PIP: ' + 
      padLeadingZeros(index2.toFixed(0), 3)  +
      ' DIP: ' + 
      padLeadingZeros(index3.toFixed(0), 3)  + ' deg';
    document.getElementById('Middle').innerHTML = 
      'M. MCP: ' + 
      padLeadingZeros(middle1.toFixed(0), 3)  +
      ' PIP: ' + 
      padLeadingZeros(middle2.toFixed(0), 3)  +
      ' DIP: ' + 
      padLeadingZeros(middle3.toFixed(0), 3)  + ' deg';  
    document.getElementById('Ring'  ).innerHTML = 
      'R. MCP: ' + 
      padLeadingZeros(ring1.toFixed(0), 3)  +
      ' PIP: ' + 
      padLeadingZeros(ring2.toFixed(0), 3)  +
      ' DIP: ' + 
      padLeadingZeros(ring3.toFixed(0), 3)  + ' deg';
    document.getElementById('Little').innerHTML = 
      'L. MCP: ' + 
      padLeadingZeros(little1.toFixed(0), 3)  +
      ' PIP: ' + 
      padLeadingZeros(little2.toFixed(0), 3)  +
      ' DIP: ' + 
      padLeadingZeros(little3.toFixed(0), 3)  + ' deg';    
    
    // Display progress bar
    progress1 = ((thumb2 + thumb3)/180*100).toFixed(0);
    document.getElementById("pThumb").style.width = progress1 + "%";
    document.getElementById("pThumb").innerHTML   = progress1 + "%";
    progress2 = ((index1 + index2 + index3)/270*100).toFixed(0);
    document.getElementById("pIndex").style.width = progress2 + "%";
    document.getElementById("pIndex").innerHTML   = progress2 + "%";
    progress3 = ((middle1 + middle2 + middle3)/270*100).toFixed(0);
    document.getElementById("pMiddle").style.width = progress3 + "%";
    document.getElementById("pMiddle").innerHTML   = progress3 + "%";
    progress4 = ((ring1 + ring2 + ring3)/270*100).toFixed(0);
    document.getElementById("pRing").style.width = progress4 + "%";
    document.getElementById("pRing").innerHTML   = progress4 + "%";
    progress5 = ((little1 + little2 + little3)/270*100).toFixed(0);
    document.getElementById("pLittle").style.width = progress5 + "%";
    document.getElementById("pLittle").innerHTML   = progress5 + "%";

    // Play sound
    if(enableSound) {
      if((thumb3 > soundThres) && (!play1)) {
        sound1.currentTime = 0;
        sound1.play();
        play1 = true;
      }
      if(thumb3 <= soundThres) {
        play1 = false;
      }
      if((index1 > soundThres) && (!play2)) {
        sound2.currentTime = 0;
        sound2.play();
        play2 = true;
      }
      if(index1 <= soundThres) {
        play2 = false;
      }
      if((middle1 > soundThres) && (!play3)) {
        sound3.currentTime = 0;
        sound3.play();
        play3 = true;
      }
      if(middle1 <= soundThres) {
        play3 = false;
      }
      if((ring1 > soundThres) && (!play4)) {
        sound4.currentTime = 0;
        sound4.play();
        play4 = true;
      }
      if(ring1 <= soundThres) {
        play4 = false;
      }
      if((little1 > soundThres) && (!play5)) {
        sound5.currentTime = 0;
        sound5.play();
        play5 = true;
      }
      if(little1 <= soundThres) {
        play5 = false;
      }
    }
  }
  canvasCtx.restore();
}

const hands = new Hands({locateFile: (file) => {
  return `https://cdn.jsdelivr.net/npm/@mediapipe/hands@0.1/${file}`;
}});
hands.onResults(onResults);

/**
 * Instantiate a camera. We'll feed each frame we receive into the solution.
 */
const camera = new Camera(videoElement, {
  onFrame: async () => {
    await hands.send({image: videoElement});
  },
  width: 1280,
  height: 720
});
camera.start();

// Present a control panel through which the user can manipulate the solution
// options.
new ControlPanel(controlsElement, {
      selfieMode: true,
      maxNumHands: 1,
      minDetectionConfidence: 0.5,
      minTrackingConfidence: 0.5,
      pauseCamera: false,
      playSound: false,
    })
    .add([
      new StaticText({title: 'Hand flexion angle'}),
      fpsControl,
      new Toggle({title: 'Selfie Mode', field: 'selfieMode'}),
      new Toggle({ title: "Pause Camera", field: "pauseCamera" }),
      new Toggle({ title: "Play Sound", field: "playSound" }),
      // new Slider(
      //     {title: 'Max Number of Hands', field: 'maxNumHands', range: [1, 4], step: 1}),
      // new Slider({
      //   title: 'Min Detection Confidence',
      //   field: 'minDetectionConfidence',
      //   range: [0, 1],
      //   step: 0.01
      // }),
      // new Slider({
      //   title: 'Min Tracking Confidence',
      //   field: 'minTrackingConfidence',
      //   range: [0, 1],
      //   step: 0.01
      // }),
    ])
    .on(options => {
      videoElement.classList.toggle('selfie', options.selfieMode);
      hands.setOptions(options);
      // Add option to pause camera
      options.pauseCamera ? camera.video.pause() : camera.start();
      // Add option to play sound
      options.playSound ? enableSound=true : enableSound=false;      
    });

// Added by GM
function getFlexionAng(results, j0, j1, j2) {
  // j0, j1, j2 are joint indexes of multiHandLandmarks
  // 1st: rescale landmarks from relative coor to absolute 3D coor
  var A = rescale(results.multiHandLandmarks[0][j0]);
  var B = rescale(results.multiHandLandmarks[0][j1]);
  var C = rescale(results.multiHandLandmarks[0][j2]);
  // 2nd: Find the acute angle at joint j1
  return getAngBtw3Pts(A, B, C);
}

function rescale(lm) {
  // Convert landmarks from relative coor to absolute 3D coor
  // Note: Assume image width:1280, height:720
  var A = {x:0, y:0, z:0};
  A.x = lm.x * 1280 - 640;
  A.y = lm.y * 720 - 360;
  A.z = lm.z * 1280;
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

function padLeadingZeros(num, size) {
  // Adapted from https://www.codegrepper.com/code-examples/javascript/convert+to+number+keep+leading+zeros+javascript
  var s = num+"";
  while (s.length < size) s = "0" + s;
  return s;
}