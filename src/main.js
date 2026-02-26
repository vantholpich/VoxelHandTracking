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
let isPinching = false;
let lastBuildTime = 0;
let lastPlacedPos = new THREE.Vector3();
let lastPinchWorldPos = new THREE.Vector3();
let results = undefined;
let handMarkers = []; // 3D spheres for landmarks
let handLines = []; // lines for connections

// --- Elements ---
const videoElement = document.getElementById('webcam');
const threeContainer = document.getElementById('three-container');
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

  // Initial hand markers pool
  const markerGeo = new THREE.SphereGeometry(0.05, 8, 8);
  const markerMat = new THREE.MeshBasicMaterial({ color: 0x00ffff });
  for (let i = 0; i < 21; i++) {
    const marker = new THREE.Mesh(markerGeo, markerMat);
    marker.visible = false;
    scene.add(marker);
    handMarkers.push(marker);
  }

  // Hand lines pool
  const lineMat = new THREE.LineBasicMaterial({ color: 0x00ffff, transparent: true, opacity: 0.5 });
  HAND_CONNECTIONS.forEach(() => {
    const geometry = new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(), new THREE.Vector3()]);
    const line = new THREE.Line(geometry, lineMat);
    line.visible = false;
    scene.add(line);
    handLines.push(line);
  });

  clock = new THREE.Clock();

  window.addEventListener('resize', onWindowResize);
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

function onWindowResize() {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
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
  // Hide all first
  handMarkers.forEach(m => m.visible = false);
  handLines.forEach(l => l.visible = false);

  if (results.landmarks && results.landmarks.length > 0) {
    const landmarks = results.landmarks[0];

    // Update marker positions
    landmarks.forEach((landmark, i) => {
      const marker = handMarkers[i];

      // Mirror X
      const vector = new THREE.Vector3(
        -((landmark.x * 2) - 1), // INVERTED for mirrored view
        - (landmark.y * 2) + 1,
        0.5
      );

      vector.unproject(camera);
      const dir = vector.clone().sub(camera.position).normalize();
      // Smoother depth estimation
      const distance = 8 - (landmark.z * 12);
      const pos = camera.position.clone().add(dir.multiplyScalar(distance));

      marker.position.copy(pos);
      marker.visible = true;
    });

    // Update line positions
    HAND_CONNECTIONS.forEach((connection, i) => {
      const line = handLines[i];
      const start = handMarkers[connection[0]].position;
      const end = handMarkers[connection[1]].position;

      const points = [start, end];
      line.geometry.setFromPoints(points);
      line.visible = true;
    });
  }
}

function processPinch(results) {
  if (!results.landmarks || results.landmarks.length === 0) {
    previewVoxel.visible = false;
    return;
  }

  const landmarks = results.landmarks[0];
  const thumbTip = landmarks[4];
  const indexTip = landmarks[8];

  const distance = Math.sqrt(
    Math.pow(thumbTip.x - indexTip.x, 2) +
    Math.pow(thumbTip.y - indexTip.y, 2) +
    Math.pow(thumbTip.z - indexTip.z, 2)
  );

  // NDC coordinates of the pinch center
  const pinchX = (thumbTip.x + indexTip.x) / 2;
  const pinchY = (thumbTip.y + indexTip.y) / 2;
  // Mirrored X for NDC
  const ndcX = -((pinchX * 2) - 1);
  const ndcY = - (pinchY * 2) + 1;

  // Use raycaster for establishing the FIRST voxel in a sequence 
  // or for the preview before pinching
  const raycaster = new THREE.Raycaster();
  raycaster.setFromCamera({ x: ndcX, y: ndcY }, camera);
  const intersects = raycaster.intersectObjects(voxels);

  let targetVoxelPos = null;
  if (intersects.length > 0) {
    const intersect = intersects[0];
    const p = intersect.object.position.clone();
    p.add(intersect.face.normal.clone().multiplyScalar(VOXEL_SIZE));
    targetVoxelPos = p;
  }

  // World position of the physical pinch for movement tracking
  const m4 = handMarkers[4].position;
  const m8 = handMarkers[8].position;
  if (!m4 || !m8) return; // Guard
  const currentPinchWorldPos = new THREE.Vector3().addVectors(m4, m8).multiplyScalar(0.5);

  if (distance < PINCH_THRESHOLD) {
    const now = performance.now();

    if (!isPinching) {
      // --- START OF NEW PINCH ---
      isPinching = true;

      let firstPos = null;
      if (voxels.length === 0) {
        // Absolute first voxel in scene
        firstPos = new THREE.Vector3(
          Math.round(currentPinchWorldPos.x / VOXEL_SIZE) * VOXEL_SIZE,
          Math.round(currentPinchWorldPos.y / VOXEL_SIZE) * VOXEL_SIZE,
          Math.round(currentPinchWorldPos.z / VOXEL_SIZE) * VOXEL_SIZE
        );
      } else if (targetVoxelPos) {
        // Starting on an existing structure
        firstPos = targetVoxelPos;
      }

      if (firstPos) {
        addVoxel(firstPos);
        lastPlacedPos.copy(firstPos);
        lastPinchWorldPos.copy(currentPinchWorldPos);
        lastBuildTime = now;
      }
    } else if (now - lastBuildTime > BUILD_COOLDOWN) {
      // --- CONTINUING PINCH (DRAGGING) ---
      const delta = currentPinchWorldPos.clone().sub(lastPinchWorldPos);
      const moveDist = delta.length();

      // If hand has moved significantly (at least half a voxel size in world space)
      if (moveDist >= VOXEL_SIZE * 0.5) {
        // Calculate next position based on dominant axis of movement from lastPinchWorldPos
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

        // Add if not already occupied
        if (!voxels.some(v => v.position.distanceTo(nextVoxelPos) < 0.1)) {
          addVoxel(nextVoxelPos);
          lastPlacedPos.copy(nextVoxelPos);
          lastPinchWorldPos.copy(currentPinchWorldPos);
          lastBuildTime = now;
        }
      }
    }

    statusElement.innerText = "Building Line...";
    statusElement.style.background = "rgba(0, 255, 0, 0.4)";
    previewVoxel.visible = false;
  } else {
    // --- NOT PINCHING ---
    if (distance > PINCH_THRESHOLD + 0.01) {
      isPinching = false;
    }

    // Show preview for where the sequence WOULD start
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

    if (previewVoxel.visible) {
      statusElement.innerText = "Targeting...";
      statusElement.style.background = "rgba(255, 255, 255, 0.1)";
    } else {
      statusElement.innerText = "Hover over existing voxel";
    }
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
