import dotenv from "dotenv";
dotenv.config();

import {
  addSensorData,
  getAllSensorData
} from "./src/components/FirebaseService/firebaseService.js";

(async () => {
  try {
    const samples = [
      { temperature: 27.1, tds: 350, turbidity: 4.1, ph: 6.8 },
      { temperature: 28.3, tds: 420, turbidity: 5.0, ph: 7.1 },
      { temperature: 26.9, tds: 380, turbidity: 3.9, ph: 6.7 }
    ];

    console.log("Adding sample data...");

    await Promise.all(samples.map(s => addSensorData(s)));

    console.log("✔ Sample sensor data added");

    const data = await getAllSensorData();

    console.log("\n📡 All Sensor Data From Firebase:");
    console.log(data);

  } catch (err) {
    console.error("❌ Error:", err);
  }
})();