import * as THREE from 'three';
import { HandLandmarker, FilesetResolver } from '@mediapipe/tasks-vision';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls';

// --- Configuration ---
const VOXEL_SIZE = 1;
const GRID_SIZE = 20;
const PINCH_THRESHOLD = 0.05; // Slightly relaxed threshold
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

  // Selection Highlight
  const highlightGeo = new THREE.PlaneGeometry(VOXEL_SIZE * 1.05, VOXEL_SIZE * 1.05);
  const highlightMat = new THREE.MeshStandardMaterial({
    color: 0x00ffff,
    transparent: true,
    opacity: 0.5,
    side: THREE.DoubleSide,
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
    updateHandMarkers(results);
    processPinch(results);
  }

  renderer.render(scene, camera);
}

function updateHandMarkers(results) {
  // Clear 2D Canvas
  ctx2d.clearRect(0, 0, canvas2d.width, canvas2d.height);

  if (results.landmarks && results.landmarks.length > 0) {
    results.landmarks.forEach((landmarks, handIdx) => {
      const handedness = results.handedness[handIdx][0];
      const isLeft = handedness ? (handedness.categoryName === "Left" || handedness.label === "Left") : false;
      const color = isLeft ? 'rgba(255, 0, 255, 0.6)' : 'rgba(0, 255, 255, 0.6)';
      const fillColor = isLeft ? '#ff00ff' : '#00ffff';

      // 1. Draw 2D Hand Connections
      ctx2d.strokeStyle = color;
      ctx2d.lineWidth = 2;
      HAND_CONNECTIONS.forEach(([startIdx, endIdx]) => {
        const start = landmarks[startIdx];
        const end = landmarks[endIdx];

        ctx2d.beginPath();
        ctx2d.moveTo(start.x * canvas2d.width, start.y * canvas2d.height);
        ctx2d.lineTo(end.x * canvas2d.width, end.y * canvas2d.height);
        ctx2d.stroke();
      });

      // 2. Draw 2D Hand Landmarks (points)
      ctx2d.fillStyle = fillColor;
      landmarks.forEach((landmark) => {
        ctx2d.beginPath();
        ctx2d.arc(landmark.x * canvas2d.width, landmark.y * canvas2d.height, 4, 0, Math.PI * 2);
        ctx2d.fill();
      });

      // 3. Update Invisible 3D markers for depth/building logic
      landmarks.forEach((landmark, i) => {
        const markerIdx = handIdx * 21 + i;
        const marker = handMarkers[markerIdx];

        // Mirror X for AR alignment
        const vector = new THREE.Vector3(
          -((landmark.x * 2) - 1),
          - (landmark.y * 2) + 1,
          0.5
        );

        vector.unproject(camera);
        const dir = vector.clone().sub(camera.position).normalize();
        const distance = 8 - (landmark.z * 12);
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

function processPinch(results) {
  if (!results.landmarks || results.landmarks.length === 0) {
    previewVoxel.visible = false;
    selectionHighlight.visible = false;
    handCursors.forEach(c => c.visible = false);
    isLeftGestureActive = false;
    return;
  }

  // Track if any hand is providing building feedback (for status UI)
  let isBuildingHandDetected = false;
  let isRotatingHandDetected = false;

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
    const currentPinchWorldPos = new THREE.Vector3().addVectors(m4, m8).multiplyScalar(0.5);

    // Update 3D Cursor to follow index tip (landmark 8)
    const cursor = handCursors[handIdx];
    if (cursor) {
      cursor.position.copy(m8); // Landmark 8 is the index tip
      cursor.material.color.set(isLeft ? 0xff00ff : 0x00ffff);
      cursor.visible = true;
    }

    if (isLeft) {
      isRotatingHandDetected = true;
      const clenched = isFist(landmarks);
      const pointing = isPointing(landmarks);

      if (clenched || pointing) {
        if (!isLeftGestureActive) {
          isLeftGestureActive = true;
          lastLeftHandPos.set(thumbTip.x, thumbTip.y);
        } else {
          // Calculate movement delta in normalized screen space
          const deltaX = thumbTip.x - lastLeftHandPos.x;
          const deltaY = thumbTip.y - lastLeftHandPos.y;
          const sensitivity = 15.0;

          if (pointing) {
            // Pointing rotates Left/Right
            controls.rotateLeft(-deltaX * sensitivity);
          }
          if (clenched) {
            // Fist rotates Up/Down
            controls.rotateUp(deltaY * sensitivity);
          }
          controls.update();

          lastLeftHandPos.set(thumbTip.x, thumbTip.y);
        }
      } else {
        isLeftGestureActive = false;
      }
    } else {
      // --- RIGHT HAND LOGIC: BUILDING ---
      isBuildingHandDetected = true;
      let targetVoxelPos = null;

      if (voxels.length > 0) {
        // Raycasting from camera through index tip (m8)
        const rayDir = m8.clone().sub(camera.position).normalize();
        raycaster.set(camera.position, rayDir);

        const intersects = raycaster.intersectObjects(voxels);

        if (intersects.length > 0) {
          const hit = intersects[0];
          const closestVoxel = hit.object;
          const normal = hit.face.normal.clone().applyQuaternion(closestVoxel.quaternion);

          targetVoxelPos = closestVoxel.position.clone().add(normal.clone().multiplyScalar(VOXEL_SIZE));
          selectionHighlight.position.copy(closestVoxel.position).add(normal.clone().multiplyScalar(VOXEL_SIZE * 0.51));
          selectionHighlight.quaternion.setFromUnitVectors(new THREE.Vector3(0, 0, 1), normal);
          selectionHighlight.visible = true;
        } else {
          selectionHighlight.visible = false;
          targetVoxelPos = null;
        }
      } else {
        selectionHighlight.visible = false;
      }

      if (distance < PINCH_THRESHOLD) {
        const now = performance.now();
        if (!isPinching) {
          isPinching = true;
          let firstPos = null;
          if (voxels.length === 0) {
            firstPos = new THREE.Vector3(
              Math.round(currentPinchWorldPos.x / VOXEL_SIZE) * VOXEL_SIZE,
              Math.round(currentPinchWorldPos.y / VOXEL_SIZE) * VOXEL_SIZE,
              Math.round(currentPinchWorldPos.z / VOXEL_SIZE) * VOXEL_SIZE
            );
          } else if (targetVoxelPos) {
            firstPos = targetVoxelPos;
          }
          if (firstPos) {
            addVoxel(firstPos);
            lastPlacedPos.copy(firstPos);
            lastPinchWorldPos.copy(currentPinchWorldPos);
            lastBuildTime = now;
          }
        } else if (now - lastBuildTime > BUILD_COOLDOWN) {
          const delta = currentPinchWorldPos.clone().sub(lastPinchWorldPos);
          const moveDist = delta.length();
          if (moveDist >= VOXEL_SIZE * 0.5) {
            const absX = Math.abs(delta.x);
            const absY = Math.abs(delta.y);
            const absZ = Math.abs(delta.z);
            const nextVoxelPos = lastPlacedPos.clone();
            if (absX >= absY && absX >= absZ) {
              nextVoxelPos.x += (delta.x > 0 ? 1 : -1) * VOXEL_SIZE;
            } else if (absY >= absX && absY >= absZ) {
              nextVoxelPos.y += (delta.y > 0 ? 1 : -1) * VOXEL_SIZE;
            } else {
              nextVoxelPos.z += (delta.z > 0 ? 1 : -1) * VOXEL_SIZE;
            }
            if (!voxels.some(v => v.position.distanceTo(nextVoxelPos) < 0.1)) {
              addVoxel(nextVoxelPos);
              lastPlacedPos.copy(nextVoxelPos);
              lastPinchWorldPos.copy(currentPinchWorldPos);
              lastBuildTime = now;
            }
          }
        }
        previewVoxel.visible = false;
      } else {
        if (distance > PINCH_THRESHOLD + 0.01) isPinching = false;
        if (voxels.length === 0) {
          previewVoxel.position.set(
            Math.round(currentPinchWorldPos.x / VOXEL_SIZE) * VOXEL_SIZE,
            Math.round(currentPinchWorldPos.y / VOXEL_SIZE) * VOXEL_SIZE,
            Math.round(currentPinchWorldPos.z / VOXEL_SIZE) * VOXEL_SIZE
          );
          previewVoxel.visible = true;
        } else if (targetVoxelPos) {
          previewVoxel.position.copy(targetVoxelPos);
          previewVoxel.visible = true;
        } else {
          previewVoxel.visible = false;
        }
      }
    }
  });

  // Update Status UI
  if (isPinching) {
    statusElement.innerText = "Building Line...";
    statusElement.style.background = "rgba(0, 255, 0, 0.4)";
  } else if (isLeftGestureActive) {
    statusElement.innerText = "Rotating View...";
    statusElement.style.background = "rgba(255, 0, 255, 0.4)";
  } else if (previewVoxel.visible) {
    statusElement.innerText = "Targeting...";
    statusElement.style.background = "rgba(255, 255, 255, 0.1)";
  } else if (isRotatingHandDetected && isBuildingHandDetected) {
    statusElement.innerText = "Left: Point to Spin | Fist to Tilt";
  } else if (isBuildingHandDetected) {
    statusElement.innerText = "Hover right hand over voxel to target";
  } else if (isRotatingHandDetected) {
    statusElement.innerText = "Point/Fist and move left hand to rotate";
  } else {
    statusElement.innerText = "Waiting for hands...";
  }

  // Cleanup visibility if hands are missing
  if (!isBuildingHandDetected) {
    previewVoxel.visible = false;
    selectionHighlight.visible = false;
  }
  // Hide inactive cursors
  handCursors.forEach((cursor, idx) => {
    if (!results.landmarks[idx]) cursor.visible = false;
  });

  if (!isRotatingHandDetected) {
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
