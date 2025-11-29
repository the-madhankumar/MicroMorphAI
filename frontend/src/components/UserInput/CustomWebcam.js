import Webcam from "react-webcam";
import { useCallback, useRef } from "react";

export const CustomWebcam = ({ onCapture }) => {
  const webcamRef = useRef(null);

  const CAPTURE_WIDTH = 700;   
  const CAPTURE_HEIGHT = 400;  

  const videoConstraints = {
    width: CAPTURE_WIDTH,
    height: CAPTURE_HEIGHT,
    facingMode: "environment",
  };

  const capture = useCallback(() => {
    if (!webcamRef.current) return;

    const imageSrc = webcamRef.current.getScreenshot({
      width: CAPTURE_WIDTH,
      height: CAPTURE_HEIGHT,
    });

    console.log("Captured image:", imageSrc);

    if (imageSrc) {
      onCapture(imageSrc);
    }
  }, [onCapture]);

  return (
    <div className="upload live-box">
      <Webcam
        ref={webcamRef}
        audio={false}
        className="webcam-view"
        screenshotFormat="image/jpeg"
        forceScreenshotSourceSize={true}
        videoConstraints={videoConstraints}
      />

      <button type="button" className="capture-btn" onClick={capture}>
        Capture
      </button>
    </div>
  );
};
