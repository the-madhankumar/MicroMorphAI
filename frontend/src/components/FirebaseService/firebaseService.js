import { initializeApp } from "firebase/app";
import { getDatabase, ref, push, set, get, update, remove, onValue } from "firebase/database";
import firebaseConfig from "./config.mjs";

const app = initializeApp(firebaseConfig);
const db = getDatabase(app);

// ---------------- Sensor Data ----------------
export function addSensorData(data) {
  const newRef = push(ref(db, "sensor_data"));
  return set(newRef, {
    temperature: data.temperature,
    tds: data.tds,
    turbidity: data.turbidity,
    ph: data.ph,
    latitude: 25.3356491,
    longitude: 83.0076292,
    timestamp: Date.now()
  });
}

export async function getAllSensorData() {
  const s = await get(ref(db, "sensor_data"));
  return s.exists() ? s.val() : {};
}

export async function getSensorDataById(id) {
  const s = await get(ref(db, `sensor_data/${id}`));
  return s.exists() ? s.val() : null;
}

export function updateSensorData(id, updatedData) {
  return update(ref(db, `sensor_data/${id}`), updatedData);
}

export function deleteSensorData(id) {
  return remove(ref(db, `sensor_data/${id}`));
}

export function listenSensorData(callback) {
  onValue(ref(db, "sensor_data"), snap => {
    callback(snap.exists() ? snap.val() : {});
  });
}

// ---------------- Inference Images ----------------
export async function getAllInferenceImages() {
  const s = await get(ref(db, "inference_images"));
  return s.exists() ? s.val() : {};
}

export async function getInferenceImageById(id) {
  const s = await get(ref(db, `inference_images/${id}`));
  return s.exists() ? s.val() : null;
}

export function listenInferenceImages(callback) {
  onValue(ref(db, "inference_images"), snap => {
    callback(snap.exists() ? snap.val() : {});
  });
}

// ---------------- Class Counts ----------------
export async function getClassCounts() {
  const s = await get(ref(db, "class_counts"));
  return s.exists() ? s.val() : {};
}

export function listenClassCounts(callback) {
  onValue(ref(db, "class_counts"), snap => {
    callback(snap.exists() ? snap.val() : {});
  });
}

export { db };
