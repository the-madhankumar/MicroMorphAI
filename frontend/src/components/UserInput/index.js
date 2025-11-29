import { Component } from "react";
import { MonitorUp } from "lucide-react";
import { CustomWebcam } from "./CustomWebcam";
import { StepForward } from 'lucide-react';
import { withRouter } from '../../withRouter';

import "./index.css";

class UserInput extends Component {
    state = {
        live: true,
        upload: false,
        imgSrc: null
    };

    handleCapture = (img) => {
        this.setState({ imgSrc: img });
    };

    handleUpload = (e) => {
        const file = e.target.files[0];
        if (file) {
            this.setState({
                imgSrc: URL.createObjectURL(file)
            });
        }
    };

    handleNextButton = () => {
        this.props.navigate('/showimages');
    }

    toggleLive = () => {
        this.setState({
            live: true,
            upload: false,
            imgSrc: null
        });
    };

    toggleUpload = () => {
        this.setState({
            live: false,
            upload: true,
            imgSrc: null
        });
    };

    render() {
        const { live, upload, imgSrc } = this.state;

        return (
            <div className="UserInput-container">
                <form className="form-container">

                    <div className="input-row">
                        <div className="input-group">
                            <label>Longitude</label>
                            <input type="number" placeholder="Enter your Longitude" />
                        </div>

                        <div className="input-group">
                            <label>Latitude</label>
                            <input type="number" placeholder="Enter your Latitude" />
                        </div>
                    </div>

                    <div className="input-row">
                        <div className="input-group">
                            <label>Depth</label>
                            <input type="number" placeholder="Enter the depth" />
                        </div>

                        <div className="input-group">
                            <label>Select Microscope</label>
                            <select>
                                <option>Choose Microscope</option>
                                <option>ISIIS</option>
                                <option>ZooScan</option>
                                <option>FlowCam</option>
                            </select>
                        </div>
                    </div>

                    <div className="input-row">
                        <div className="input-group">
                            <label>Date</label>
                            <input type="date" />
                        </div>

                        <div className="input-group">
                            <label>Time</label>
                            <input type="time" />
                        </div>
                    </div>

                    <div className="input-row">
                        <div className="input-group full-width">
                            <label>Ship Name</label>
                            <input type="text" placeholder="Enter the Ship Name" />
                        </div>
                    </div>

                    <div className="input-row">
                        {live && (
                            <div className="input-group full-width">
                                <label>Live Image</label>
                                <CustomWebcam onCapture={this.handleCapture} />
                            </div>
                        )}

                        {upload && (
                            <div className="input-group full-width">
                                <label>Upload Image</label>

                                <label className="upload">
                                    <MonitorUp size={40} />
                                    <input
                                        type="file"
                                        accept="image/*"
                                        onChange={this.handleUpload}
                                        className="upload-input"
                                    />
                                </label>
                            </div>
                        )}
                    </div>

                    {imgSrc && (
                        <img src={imgSrc} alt="preview" className="uploaded-preview" />
                    )}

                    <div className="input-row">
                        <button type="button" onClick={this.toggleLive} className="cus-button">
                            Live
                        </button>

                        <button type="button" onClick={this.toggleUpload} className="cus-button">
                            Upload
                        </button>
                    </div>
                    <button type="button" className="custom-next-button" onClick={this.handleNextButton}>
                        <span className="btn-text">Next</span>
                        <span className="btn-file"><StepForward /></span>
                    </button>
                </form>
            </div>
        );
    }
}

export default withRouter(UserInput);
