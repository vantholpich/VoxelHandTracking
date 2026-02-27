import * as THREE from 'three';
import { HandLandmarker, FilesetResolver } from '@mediapipe/tasks-vision';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls';

// --- Configuration ---
const VOXEL_SIZE = 1;
const GRID_SIZE = 20;
const PINCH_THRESHOLD = 0.045; // Slightly relaxed for reliability
const BUILD_COOLDOWN = 300; // Increased cooldown to prevent accidental multiple builds

const HAND_CONNECTIONS = [
  [0, 1], [1, 2], [2, 3], [3, 4], // Thumb
  [0, 5], [5, 6], [6, 7], [7, 8], // Index
  [0, 9], [9, 10], [10, 11], [11, 12], // Middle
  [0, 13], [13, 14], [14, 15], [15, 16], // Ring
  [0, 17], [17, 18], [18, 19], [19, 20], // Pinky
  [5, 9], [9, 13], [13, 17] // Palm
];

// --- State ---
let scene, camera, renderer, clock, controls;
let handLandmarker;
let webcam;
let lastVideoTime = -1;
let voxels = [];
let previewVoxel;
let selectionHighlight;
const raycaster = new THREE.Raycaster();
let isPinching = false;
let lastBuildTime = 0;
let lastPlacedPos = new THREE.Vector3();
let lastPinchWorldPos = new THREE.Vector3();
let results = undefined;
let handMarkers = []; // 3D spheres for landmarks
let handLines = []; // lines for connections
let handCursors = []; // 3D cursors for each hand
let isLeftGestureActive = false;
let lastLeftHandPos = new THREE.Vector2();
let currentBuildNormal = null;
let initialBuildPos = new THREE.Vector3();
let lastLeftPinchDistance = 0; // State for zoom gesture

// --- Elements ---
const videoElement = document.getElementById('webcam');
const threeContainer = document.getElementById('three-container');
const canvas2d = document.getElementById('gesture-canvas');
const ctx2d = canvas2d.getContext('2d');
const statusElement = document.getElementById('status');

// --- Initialization ---

async function init() {
  setupThree();
  await setupHandTracking();
  setupWebcam();
  animate();
}

function setupThree() {
  scene = new THREE.Scene();
  // Transparent background to show video
  // scene.background = new THREE.Color(0x0a0a0a); 

  camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
  camera.position.set(0, 0, 10);
  camera.lookAt(0, 0, 0);

  renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  renderer.setClearColor(0x000000, 0); // Transparent
  renderer.setSize(window.innerWidth, window.innerHeight);
  renderer.setPixelRatio(window.devicePixelRatio);
  threeContainer.appendChild(renderer.domElement);

  controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.05;
  controls.screenSpacePanning = false;
  controls.minDistance = 5;
  controls.maxDistance = 50;
  controls.maxPolarAngle = Math.PI / 2;

  // Lights
  const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
  scene.add(ambientLight);

  const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
  directionalLight.position.set(5, 10, 7);
  scene.add(directionalLight);

  // Removed Grid and Ground Plane

  // Preview Voxel
  const previewGeo = new THREE.BoxGeometry(VOXEL_SIZE, VOXEL_SIZE, VOXEL_SIZE);
  const previewMat = new THREE.MeshStandardMaterial({
    color: 0x00ff00,
    transparent: true,
    opacity: 0.5,
    wireframe: true
  });
  previewVoxel = new THREE.Mesh(previewGeo, previewMat);
  previewVoxel.visible = false;
  scene.add(previewVoxel);

  // Selection Highlight (Box surround)
  const highlightGeo = new THREE.BoxGeometry(VOXEL_SIZE * 1.05, VOXEL_SIZE * 1.05, VOXEL_SIZE * 1.05);
  const highlightMat = new THREE.MeshStandardMaterial({
    color: 0x00ffff,
    transparent: true,
    opacity: 0.3,
    wireframe: true,
    emissive: 0x00ffff,
    emissiveIntensity: 0.5
  });
  selectionHighlight = new THREE.Mesh(highlightGeo, highlightMat);
  selectionHighlight.visible = false;
  scene.add(selectionHighlight);

  // Invisible 3D markers for depth/building calculations (2 hands)
  for (let i = 0; i < 42; i++) {
    const marker = new THREE.Object3D();
    scene.add(marker);
    handMarkers.push(marker);
  }

  // Cursors (2 hands)
  for (let i = 0; i < 2; i++) {
    const cursorGeo = new THREE.SphereGeometry(0.1, 16, 16);
    const cursorMat = new THREE.MeshBasicMaterial({ color: 0xffffff });
    const cursor = new THREE.Mesh(cursorGeo, cursorMat);
    cursor.visible = false;
    scene.add(cursor);
    handCursors.push(cursor);
  }

  clock = new THREE.Clock();

  window.addEventListener('resize', onWindowResize);
  onWindowResize(); // Set initial dimensions
}

function onWindowResize() {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);

  canvas2d.width = window.innerWidth;
  canvas2d.height = window.innerHeight;
}

async function setupHandTracking() {
  statusElement.innerText = 'Loading Hand Landmarker...';
  const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm"
  );
  handLandmarker = await HandLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath: `https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task`,
      delegate: "GPU"
    },
    runningMode: "VIDEO",
    numHands: 2
  });
  statusElement.innerText = 'Hand Tracking Ready';
}

function setupWebcam() {
  navigator.mediaDevices.getUserMedia({ video: true }).then((stream) => {
    videoElement.srcObject = stream;
  });
}

// --- Logic ---

function animate() {
  requestAnimationFrame(animate);

  if (controls) controls.update();

  if (videoElement.readyState >= 2) if (videoElement.currentTime !== lastVideoTime) {
    lastVideoTime = videoElement.currentTime;
    results = handLandmarker.detectForVideo(videoElement, performance.now());

    // Calculate mapping for object-fit: cover (centered scaling)
    const videoAspect = videoElement.videoWidth / videoElement.videoHeight;
    const windowAspect = window.innerWidth / window.innerHeight;
    let scaleX = 1, scaleY = 1;

    if (videoAspect > windowAspect) {
      // Video is wider than window (cropped horizontally)
      scaleX = videoAspect / windowAspect;
    } else {
      // Video is taller than window (cropped vertically)
      scaleY = windowAspect / videoAspect;
    }

    const mapping = { scaleX, scaleY };
    updateHandMarkers(results, mapping);
    processPinch(results);
  }

  renderer.render(scene, camera);
}

function updateHandMarkers(results, mapping) {
  // Clear 2D Canvas
  ctx2d.clearRect(0, 0, canvas2d.width, canvas2d.height);

  if (results.landmarks && results.landmarks.length > 0) {
    results.landmarks.forEach((landmarks, handIdx) => {
      // Corrected landmarks for screen display (centered scaling)
      const correctedLandmarks = landmarks.map(l => ({
        x: (l.x - 0.5) * mapping.scaleX + 0.5,
        y: (l.y - 0.5) * mapping.scaleY + 0.5,
        z: l.z
      }));
      const handedness = results.handedness[handIdx][0];
      const isLeft = handedness ? (handedness.categoryName === "Left" || handedness.label === "Left") : false;
      const color = isLeft ? 'rgba(255, 0, 255, 0.6)' : 'rgba(0, 255, 255, 0.6)';
      const fillColor = isLeft ? '#ff00ff' : '#00ffff';

      // 1. Draw 2D Hand Connections
      ctx2d.strokeStyle = color;
      ctx2d.lineWidth = 2;
      HAND_CONNECTIONS.forEach(([startIdx, endIdx]) => {
        const start = correctedLandmarks[startIdx];
        const end = correctedLandmarks[endIdx];

        ctx2d.beginPath();
        ctx2d.moveTo(start.x * canvas2d.width, start.y * canvas2d.height);
        ctx2d.lineTo(end.x * canvas2d.width, end.y * canvas2d.height);
        ctx2d.stroke();
      });

      // 2. Draw 2D Hand Landmarks (points)
      ctx2d.fillStyle = fillColor;
      correctedLandmarks.forEach((landmark) => {
        ctx2d.beginPath();
        ctx2d.arc(landmark.x * canvas2d.width, landmark.y * canvas2d.height, 4, 0, Math.PI * 2);
        ctx2d.fill();
      });

      // 3. Update Invisible 3D markers for depth/building logic
      correctedLandmarks.forEach((landmark, i) => {
        const markerIdx = handIdx * 21 + i;
        const marker = handMarkers[markerIdx];

        // Mirror X for AR alignment (since canvas is mirrored)
        const vector = new THREE.Vector3(
          -((landmark.x * 2) - 1),
          - (landmark.y * 2) + 1,
          0.5
        );

        vector.unproject(camera);
        const dir = vector.clone().sub(camera.position).normalize();

        // --- Improved Depth Estimation via Hand Scale ---
        // Distance between wrist (0) and middle finger MCP (9)
        const wrist = landmarks[0];
        const mcp = landmarks[9];
        const handScale = Math.sqrt(Math.pow(wrist.x - mcp.x, 2) + Math.pow(wrist.y - mcp.y, 2));

        // --- Natural Depth Mapping (Direct Relationship) ---
        // Hand further from webcam (lower scale) -> Voxel closer to camera
        // Hand closer to webcam (higher scale) -> Voxel further from camera
        const baseDistance = handScale * 25 + 2;

        // Landmark.z is negative when closer to camera relative to the hand root
        const relativeDepth = landmark.z * 10;
        const distance = baseDistance + relativeDepth;

        const pos = camera.position.clone().add(dir.multiplyScalar(distance));

        marker.position.copy(pos);
      });
    });
  }
}

function isFist(landmarks) {
  const fingerTips = [8, 12, 16, 20];
  const fingerBases = [5, 9, 13, 17];
  const wrist = landmarks[0];

  let curledCount = 0;
  for (let i = 0; i < 4; i++) {
    const tip = landmarks[fingerTips[i]];
    const base = landmarks[fingerBases[i]];
    const dTip = Math.sqrt(Math.pow(tip.x - wrist.x, 2) + Math.pow(tip.y - wrist.y, 2));
    const dBase = Math.sqrt(Math.pow(base.x - wrist.x, 2) + Math.pow(base.y - wrist.y, 2));
    if (dTip < dBase) curledCount++;
  }
  return curledCount >= 4;
}

function isPointing(landmarks) {
  const fingerTips = [12, 16, 20]; // Middle, Ring, Pinky
  const fingerBases = [9, 13, 17];
  const wrist = landmarks[0];

  // Index (8) should be extended
  const indexTip = landmarks[8];
  const indexBase = landmarks[5];
  const dIndexTip = Math.sqrt(Math.pow(indexTip.x - wrist.x, 2) + Math.pow(indexTip.y - wrist.y, 2));
  const dIndexBase = Math.sqrt(Math.pow(indexBase.x - wrist.x, 2) + Math.pow(indexBase.y - wrist.y, 2));

  if (dIndexTip <= dIndexBase * 1.2) return false;

  // Others should be curled
  let curledCount = 0;
  for (let i = 0; i < 3; i++) {
    const tip = landmarks[fingerTips[i]];
    const base = landmarks[fingerBases[i]];
    const dTip = Math.sqrt(Math.pow(tip.x - wrist.x, 2) + Math.pow(tip.y - wrist.y, 2));
    const dBase = Math.sqrt(Math.pow(base.x - wrist.x, 2) + Math.pow(base.y - wrist.y, 2));
    if (dTip < dBase) curledCount++;
  }
  return curledCount >= 3;
}

function isThreeFingerPose(landmarks) {
  const wrist = landmarks[0];

  // Thumb (4) extended
  const thumbTip = landmarks[4];
  const thumbBase = landmarks[2];
  const dThumbTip = Math.sqrt(Math.pow(thumbTip.x - wrist.x, 2) + Math.pow(thumbTip.y - wrist.y, 2));
  const dThumbBase = Math.sqrt(Math.pow(thumbBase.x - wrist.x, 2) + Math.pow(thumbBase.y - wrist.y, 2));
  if (dThumbTip <= dThumbBase) return false;

  // Index (8) extended
  const indexTip = landmarks[8];
  const indexBase = landmarks[5];
  const dIndexTip = Math.sqrt(Math.pow(indexTip.x - wrist.x, 2) + Math.pow(indexTip.y - wrist.y, 2));
  const dIndexBase = Math.sqrt(Math.pow(indexBase.x - wrist.x, 2) + Math.pow(indexBase.y - wrist.y, 2));
  if (dIndexTip <= dIndexBase * 1.2) return false;

  // Middle (12) extended
  const middleTip = landmarks[12];
  const middleBase = landmarks[9];
  const dMiddleTip = Math.sqrt(Math.pow(middleTip.x - wrist.x, 2) + Math.pow(middleTip.y - wrist.y, 2));
  const dMiddleBase = Math.sqrt(Math.pow(middleBase.x - wrist.x, 2) + Math.pow(middleBase.y - wrist.y, 2));
  if (dMiddleTip <= dMiddleBase * 1.2) return false;

  // Ring (16) and Pinky (20) curled
  const curledTips = [16, 20];
  const curledBases = [13, 17];
  for (let i = 0; i < 2; i++) {
    const tip = landmarks[curledTips[i]];
    const base = landmarks[curledBases[i]];
    const dTip = Math.sqrt(Math.pow(tip.x - wrist.x, 2) + Math.pow(tip.y - wrist.y, 2));
    const dBase = Math.sqrt(Math.pow(base.x - wrist.x, 2) + Math.pow(base.y - wrist.y, 2));
    if (dTip >= dBase) return false;
  }

  return true;
}

function isTiltingPose(landmarks) {
  const wrist = landmarks[0];

  // Thumb (4) extended
  const thumbTip = landmarks[4];
  const thumbBase = landmarks[2];
  const dThumbTip = Math.sqrt(Math.pow(thumbTip.x - wrist.x, 2) + Math.pow(thumbTip.y - wrist.y, 2));
  const dThumbBase = Math.sqrt(Math.pow(thumbBase.x - wrist.x, 2) + Math.pow(thumbBase.y - wrist.y, 2));
  if (dThumbTip <= dThumbBase) return false;

  // Index (8), Middle (12), Ring (16) extended
  const extendedTips = [8, 12, 16];
  const extendedBases = [5, 9, 13];
  for (let i = 0; i < 3; i++) {
    const tip = landmarks[extendedTips[i]];
    const base = landmarks[extendedBases[i]];
    const dTip = Math.sqrt(Math.pow(tip.x - wrist.x, 2) + Math.pow(tip.y - wrist.y, 2));
    const dBase = Math.sqrt(Math.pow(base.x - wrist.x, 2) + Math.pow(base.y - wrist.y, 2));
    if (dTip <= dBase * 1.2) return false;
  }

  // Pinky (20) curled
  const pinkyTip = landmarks[20];
  const pinkyBase = landmarks[17];
  const dPinkyTip = Math.sqrt(Math.pow(pinkyTip.x - wrist.x, 2) + Math.pow(pinkyTip.y - wrist.y, 2));
  const dPinkyBase = Math.sqrt(Math.pow(pinkyBase.x - wrist.x, 2) + Math.pow(pinkyBase.y - wrist.y, 2));
  if (dPinkyTip >= dPinkyBase) return false;

  return true;
}

function isGrip(landmarks) {
  const fingerTips = [12, 16, 20]; // Middle, Ring, Pinky
  const fingerBases = [9, 13, 17];
  const wrist = landmarks[0];

  let curledCount = 0;
  for (let i = 0; i < 3; i++) {
    const tip = landmarks[fingerTips[i]];
    const base = landmarks[fingerBases[i]];
    const dTip = Math.sqrt(Math.pow(tip.x - wrist.x, 2) + Math.pow(tip.y - wrist.y, 2));
    const dBase = Math.sqrt(Math.pow(base.x - wrist.x, 2) + Math.pow(base.y - wrist.y, 2));
    if (dTip < dBase) curledCount++;
  }

  // Thumb (4) should be extended
  const thumbTip = landmarks[4];
  const thumbBase = landmarks[2];
  const dThumbTip = Math.sqrt(Math.pow(thumbTip.x - wrist.x, 2) + Math.pow(thumbTip.y - wrist.y, 2));
  const dThumbBase = Math.sqrt(Math.pow(thumbBase.x - wrist.x, 2) + Math.pow(thumbBase.y - wrist.y, 2));
  const thumbExtended = dThumbTip > dThumbBase;

  return (curledCount >= 3) && thumbExtended;
}

function processPinch(results) {
  if (!results.landmarks || results.landmarks.length === 0) {
    previewVoxel.visible = false;
    selectionHighlight.visible = false;
    handCursors.forEach(c => c.visible = false);
    isLeftGestureActive = false;
    isPinching = false;
    currentBuildNormal = null;
    return;
  }

  // Frame-level aggregate state
  let frameTargetVoxelPos = null;
  let frameTargetNormal = null;
  let isAnyRotatingHandDetected = false;
  let isAnyBuildingHandDetected = false;
  let frameCurrentPinchWorldPos = null;
  let framePinchDistance = 999;

  results.landmarks.forEach((landmarks, handIdx) => {
    const handedness = results.handedness[handIdx][0];
    const isLeft = handedness ? (handedness.categoryName === "Left" || handedness.label === "Left") : false;

    const thumbTip = landmarks[4];
    const indexTip = landmarks[8];
    const distance = Math.sqrt(
      Math.pow(thumbTip.x - indexTip.x, 2) +
      Math.pow(thumbTip.y - indexTip.y, 2) +
      Math.pow(thumbTip.z - indexTip.z, 2)
    );

    const m4 = handMarkers[handIdx * 21 + 4].position;
    const m8 = handMarkers[handIdx * 21 + 8].position;
    if (!m4 || !m8) return;
    const currentHandPinchWorldPos = new THREE.Vector3().addVectors(m4, m8).multiplyScalar(0.5);

    // Update 3D Cursor to follow index tip
    const cursor = handCursors[handIdx];
    if (cursor) {
      cursor.position.copy(m8);
      cursor.material.color.set(isLeft ? 0xff00ff : 0x00ffff);
      cursor.visible = true;
    }

    if (isLeft) {
      isAnyRotatingHandDetected = true;
      const spinning = isThreeFingerPose(landmarks);
      const tilting = isTiltingPose(landmarks);

      if (spinning || tilting) {
        if (!isLeftGestureActive) {
          isLeftGestureActive = true;
          lastLeftHandPos.set(thumbTip.x, thumbTip.y);
        } else {
          const sensitivity = 15.0;
          if (spinning) {
            const deltaX = thumbTip.x - lastLeftHandPos.x;
            controls.rotateLeft(-deltaX * sensitivity);
          }
          if (tilting) {
            const deltaY = thumbTip.y - lastLeftHandPos.y;
            controls.rotateUp(deltaY * sensitivity);
          }
          controls.update();
          lastLeftHandPos.set(thumbTip.x, thumbTip.y);
        }
      } else if (isGrip(landmarks)) {
        // --- ZOOM LOGIC (Left Hand) ---
        // Only zoom if middle, ring, pinky are curled (isGrip)
        isLeftGestureActive = false;

        // Removed distance < threshold check to allow for wide spreading
        if (lastLeftPinchDistance > 0) {
          const deltaDist = distance - lastLeftPinchDistance;
          const zoomSensitivity = 15.0; // Reduced for smoother control

          if (Math.abs(deltaDist) > 0.001) {
            // Inverting logic: User reports spread = further. 
            // We swap so spread (deltaDist > 0) -> dollyIn (closer)
            // Wait, if users says spread CURRENTLY makes it further, and my code has dollyIn...
            // Then dollyIn is making it further? That's impossible.
            // Let's swap the functions as requested.
            if (deltaDist > 0) {
              controls.dollyOut(1 + deltaDist * zoomSensitivity);
            } else {
              controls.dollyIn(1 - deltaDist * zoomSensitivity);
            }
            controls.update();
          }
        }
        lastLeftPinchDistance = distance;
      } else {
        isLeftGestureActive = false;
        lastLeftPinchDistance = 0;
      }
    } else {
      // --- RIGHT HAND LOGIC: SELECTION & PINCH DETECTION ---
      isAnyBuildingHandDetected = true;
      frameCurrentPinchWorldPos = currentHandPinchWorldPos;
      framePinchDistance = distance;

      if (voxels.length > 0) {
        const rayDir = m8.clone().sub(camera.position).normalize();
        raycaster.set(camera.position, rayDir);
        const intersects = raycaster.intersectObjects(voxels);

        if (intersects.length > 0) {
          const hit = intersects[0];
          frameTargetVoxelPos = hit.object.position.clone();
        }
      }
    }
  });

  // --- GLOBAL STATE UPDATE & BUILDING ---
  if (isAnyBuildingHandDetected) {
    // 1. Update Selection Highlight
    if (isPinching) {
      selectionHighlight.position.copy(lastPlacedPos);
      selectionHighlight.visible = true;
    } else if (frameTargetVoxelPos) {
      selectionHighlight.position.copy(frameTargetVoxelPos);
      selectionHighlight.visible = true;
    } else {
      selectionHighlight.visible = false;
    }

    // 2. Process Pinch State
    if (framePinchDistance < PINCH_THRESHOLD) {
      const now = performance.now();
      if (!isPinching) {
        let startPos = null;
        let startNormal = null;

        if (voxels.length === 0) {
          startPos = new THREE.Vector3(
            Math.round(frameCurrentPinchWorldPos.x / VOXEL_SIZE) * VOXEL_SIZE,
            Math.round(frameCurrentPinchWorldPos.y / VOXEL_SIZE) * VOXEL_SIZE,
            Math.round(frameCurrentPinchWorldPos.z / VOXEL_SIZE) * VOXEL_SIZE
          );
          addVoxel(startPos);
          lastPlacedPos.copy(startPos);
        } else if (frameTargetVoxelPos) {
          startPos = frameTargetVoxelPos;
          lastPlacedPos.copy(startPos); // Start tracking from the targeted voxel
        }

        if (startPos || frameTargetVoxelPos) {
          isPinching = true;
          currentBuildNormal = null; // Reset normal to determine it once movement starts
          lastPinchWorldPos.copy(frameCurrentPinchWorldPos);
          lastBuildTime = now;
        }
      } else if (now - lastBuildTime > BUILD_COOLDOWN) {
        // Calculate delta from last pinch pos
        const handDelta = frameCurrentPinchWorldPos.clone().sub(lastPinchWorldPos);

        if (currentBuildNormal) {
          // Direction is locked - build only along this axis
          const projectedDist = handDelta.dot(currentBuildNormal);
          if (Math.abs(projectedDist) >= VOXEL_SIZE * 0.6) {
            const steps = Math.sign(projectedDist);
            const nextVoxelPos = lastPlacedPos.clone().add(currentBuildNormal.clone().multiplyScalar(steps * VOXEL_SIZE));

            if (!voxels.some(v => v.position.distanceTo(nextVoxelPos) < 0.1)) {
              addVoxel(nextVoxelPos);
              lastPlacedPos.copy(nextVoxelPos);
              lastPinchWorldPos.copy(frameCurrentPinchWorldPos);
              lastBuildTime = now;
            }
          }
        } else {
          // Determine the dominant axis of movement to lock it
          const absX = Math.abs(handDelta.x);
          const absY = Math.abs(handDelta.y);
          const absZ = Math.abs(handDelta.z);
          const maxDelta = Math.max(absX, absY, absZ);

          if (maxDelta >= VOXEL_SIZE * 1.2) { // Increased confidence threshold for initial lock
            let moveDir = new THREE.Vector3();
            if (maxDelta === absX) moveDir.set(Math.sign(handDelta.x), 0, 0);
            else if (maxDelta === absY) moveDir.set(0, Math.sign(handDelta.y), 0);
            else moveDir.set(0, 0, Math.sign(handDelta.z));

            const nextVoxelPos = lastPlacedPos.clone().add(moveDir.clone().multiplyScalar(VOXEL_SIZE));

            if (!voxels.some(v => v.position.distanceTo(nextVoxelPos) < 0.1)) {
              addVoxel(nextVoxelPos);
              lastPlacedPos.copy(nextVoxelPos);
              lastPinchWorldPos.copy(frameCurrentPinchWorldPos);
              lastBuildTime = now;
              currentBuildNormal = moveDir.clone(); // LOCK the direction
            }
          }
        }
      }
      previewVoxel.visible = false;
    } else {
      if (framePinchDistance > PINCH_THRESHOLD + 0.01) {
        isPinching = false;
        currentBuildNormal = null;
      }
      // 3. Update Preview Voxel
      if (voxels.length === 0) {
        previewVoxel.position.set(
          Math.round(frameCurrentPinchWorldPos.x / VOXEL_SIZE) * VOXEL_SIZE,
          Math.round(frameCurrentPinchWorldPos.y / VOXEL_SIZE) * VOXEL_SIZE,
          Math.round(frameCurrentPinchWorldPos.z / VOXEL_SIZE) * VOXEL_SIZE
        );
        previewVoxel.visible = true;
      } else if (frameTargetVoxelPos) {
        // No preview when targeting existing voxel, just the selection highlight
        previewVoxel.visible = false;
      } else {
        previewVoxel.visible = false;
      }
    }
  } else {
    previewVoxel.visible = false;
    selectionHighlight.visible = false;
  }

  // Update Status UI
  if (isPinching) {
    statusElement.innerText = "Building...";
    statusElement.style.background = "rgba(0, 255, 0, 0.4)";
  } else if (isLeftGestureActive) {
    statusElement.innerText = "Rotating View...";
    statusElement.style.background = "rgba(255, 0, 255, 0.4)";
  } else if (frameTargetVoxelPos) {
    statusElement.innerText = "Pinch and drag (Right Hand) to Build";
    statusElement.style.background = "rgba(0, 255, 255, 0.2)";
  } else if (isAnyRotatingHandDetected && isAnyBuildingHandDetected) {
    statusElement.innerText = "Left: 3-Fingers Spin | 4-Fingers Tilt | Spread/Pinch Zoom";
  } else if (isAnyBuildingHandDetected) {
    statusElement.innerText = "Hover right hand over a block to start building";
  } else if (isAnyRotatingHandDetected) {
    statusElement.innerText = "Left: 3-Fingers Spin | 4-Fingers Tilt | Spread/Pinch Zoom";
  } else {
    statusElement.innerText = "Waiting for hands...";
  }

  // Hide inactive cursors
  handCursors.forEach((cursor, idx) => {
    if (!results.landmarks[idx]) cursor.visible = false;
  });

  if (!isAnyRotatingHandDetected) {
    isLeftGestureActive = false;
  }
}

function addVoxel(pos) {
  // Check if a voxel already exists at this exact position
  const exists = voxels.some(v => v.position.distanceTo(pos) < 0.1);
  if (exists) return;

  const geo = new THREE.BoxGeometry(VOXEL_SIZE, VOXEL_SIZE, VOXEL_SIZE);
  const mat = new THREE.MeshStandardMaterial({
    color: new THREE.Color().setHSL(Math.random(), 0.7, 0.6),
    metalness: 0.2,
    roughness: 0.1,
    wireframe: true
  });
  const voxel = new THREE.Mesh(geo, mat);
  voxel.position.copy(pos);
  scene.add(voxel);
  voxels.push(voxel);
}

init();
