import { Component } from "react";
import { getAllSensorData, addSensorData } from "../FirebaseService/firebaseService.js";

import SimpleCharts from "./BarChart";
import CompositionExample from "./GaugeChart";
import showImagesList from "../../Data/showImage.js";


import BasicLineChart from "./LineChart/index.js";
import { ArrowBigRightDash, ArrowBigLeftDash } from "lucide-react";

import { MapContainer, TileLayer, Marker } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';
import L from 'leaflet';

import "./index.css";

delete L.Icon.Default.prototype._getIconUrl;

L.Icon.Default.mergeOptions({
    iconRetinaUrl: require('leaflet/dist/images/marker-icon-2x.png'),
    iconUrl: require('leaflet/dist/images/marker-icon.png'),
    shadowUrl: require('leaflet/dist/images/marker-shadow.png'),
});


class ShowImages extends Component {
    state = {
        showImageData: [],
        carouselIndex: 1,
        currentImageSpec: null,
        sensorData: null,
        sensorDataPh: [],
        sensorDatatds: [],
        sensorDataTemperature: [],
        sensorDataTurbidity: [],
        sensorDataTime: [],

        showDataPh: true,
        showDatatds: false,
        showDataTemperature: false,
        showDataTurbidity: false,
        showDataTime: false,
    };

    // async componentDidMount() {
    //     let baseTime = 0; // start at 0, not current time

    //     const samples = [
    //         { temperature: 27.5, tds: 350, turbidity: 4.0, ph: 6.8 },
    //         { temperature: 27.8, tds: 355, turbidity: 4.1, ph: 6.9 },
    //         { temperature: 28.1, tds: 360, turbidity: 4.2, ph: 7.0 },
    //         { temperature: 28.3, tds: 365, turbidity: 4.3, ph: 7.1 },
    //         { temperature: 28.6, tds: 370, turbidity: 4.4, ph: 7.2 },
    //         { temperature: 28.4, tds: 368, turbidity: 4.3, ph: 7.1 },
    //         { temperature: 28.1, tds: 362, turbidity: 4.2, ph: 7.0 },
    //         { temperature: 27.9, tds: 358, turbidity: 4.1, ph: 6.9 },
    //         { temperature: 27.6, tds: 353, turbidity: 4.0, ph: 6.8 },
    //         { temperature: 27.4, tds: 349, turbidity: 3.9, ph: 6.7 }
    //     ];

    //     const samplesWithTime = samples.map((sample, index) => ({
    //         ...sample,
    //         timestamp: baseTime + index * 1000 // 1 second interval
    //     }));

    //     samplesWithTime.forEach(sample => addSensorData(sample));
    // }


    async componentDidMount() {
        const allSensorData = await getAllSensorData();
        const arrayData = Object.values(allSensorData || {});

        const phData = arrayData.map(item => item.ph);
        const tdsData = arrayData.map(item => item.tds);
        const tempData = arrayData.map(item => item.temperature);
        const turbidityData = arrayData.map(item => item.turbidity);
        const timeData = arrayData.map(item => item.timestamp);

        this.setState({
            showImageData: showImagesList,
            currentImageSpec: showImagesList[1] || null,
            sensorData: arrayData,
            sensorDataPh: phData,
            sensorDatatds: tdsData,
            sensorDataTemperature: tempData,
            sensorDataTurbidity: turbidityData,
            sensorDataTime: timeData
        });
    }

    decrementCarouselIndex = () => {
        const { carouselIndex, showImageData } = this.state;
        const newIndex = Math.max(carouselIndex - 1, 1);
        this.setState({
            carouselIndex: newIndex,
            currentImageSpec: showImageData[newIndex]
        });
    };

    incrementCarouselIndex = () => {
        const { carouselIndex, showImageData } = this.state;
        const newIndex = Math.min(carouselIndex + 1, showImageData.length - 1);
        this.setState({
            carouselIndex: newIndex,
            currentImageSpec: showImageData[newIndex]
        });
    };

    handleShowAnalysis = (id) => {
        const keyMap = {
            1: "showDataPh",
            2: "showDataTemperature",
            3: "showDataTurbidity",
            4: "showDatatds",
            5: "showDataTime"
        };

        const key = keyMap[id];

        if (key) {
            this.setState({
                showDataPh: false,
                showDataTemperature: false,
                showDataTurbidity: false,
                showDatatds: false,
                showDataTime: false,
                [key]: true
            });
        }
    };

    render() {
        const { showImageData, carouselIndex, currentImageSpec, sensorData } = this.state;

        const mainImage = showImageData[0];
        const currentImage = showImageData[carouselIndex];

        const lastReading = sensorData && sensorData.length ? sensorData[sensorData.length - 1] : {};

        return (
            <div className="showimages-container">
                <div className="showimages-main-image-container">
                    {mainImage && (
                        <img
                            src={mainImage.image}
                            alt={mainImage.class}
                            className="show-main-image"
                        />
                    )}
                    <div className="classification-stats">
                        <h1>Classification Stats</h1>
                        <div className="main-image-specs">
                            <SimpleCharts />
                        </div>
                    </div>
                </div>

                <div className="stats-section">
                    <div className="section-1">
                        <div className="carousel-complete-view">
                            <div className="carousel-wrapper">
                                <button
                                    onClick={this.decrementCarouselIndex}
                                    className="carousel-arrow left"
                                    disabled={carouselIndex === 1}
                                >
                                    <ArrowBigLeftDash size={50} />
                                </button>

                                {currentImage && (
                                    <img
                                        src={currentImage.image}
                                        alt={currentImage.class}
                                        className="carousel-image"
                                    />
                                )}

                                <button
                                    onClick={this.incrementCarouselIndex}
                                    className="carousel-arrow right"
                                    disabled={carouselIndex === showImageData.length - 1}
                                >
                                    <ArrowBigRightDash size={50} />
                                </button>
                            </div>

                            <div className="current-image-classification">
                                <h1>Current Image Specification</h1>
                                {currentImageSpec && (
                                    <div className="image-spec-details">
                                        <p className="image-class">{currentImageSpec.class}</p>
                                        <p className="image-confidence">
                                            Confidence: {currentImageSpec.confidence || "N/A"}
                                        </p>
                                        <CompositionExample confidence={currentImageSpec.confidence} />
                                    </div>
                                )}
                            </div>
                        </div>
                    </div>

                    <div className="section-2">
                        <div className="water-analysis">
                            <div className="water-metric">pH: {lastReading.ph || "-"}</div>
                            <div className="water-metric">Temperature: {lastReading.temperature || "-"}</div>
                            <div className="water-metric">Turbidity: {lastReading.turbidity || "-"}</div>
                            <div className="water-metric">TDS: {lastReading.tds || "-"}</div>
                            <div className="water-metric">
                                GPS Location: {lastReading.latitude && lastReading.longitude
                                    ? `${lastReading.latitude}, ${lastReading.longitude}`
                                    : "-"}

                                {lastReading.latitude && lastReading.longitude && (
                                    <MapContainer
                                        center={[lastReading.latitude, lastReading.longitude]}
                                        zoom={13}
                                        scrollWheelZoom={false}
                                        style={{ height: '400px', width: '100%' }}
                                    >
                                        <TileLayer
                                            url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
                                            attribution='&copy; <a href="http://osm.org/copyright">OpenStreetMap</a> contributors'
                                        />
                                        <Marker position={[lastReading.latitude, lastReading.longitude]} />
                                    </MapContainer>
                                )}
                            </div>
                        </div>

                        <div>
                            <p className="water-header water-header-graph">Water quality trend gap</p>

                            <div className="selectAnalysis">
                                <div
                                    className={`select-box ${this.state.showDataPh ? "active" : ""}`}
                                    onClick={() => this.handleShowAnalysis(1)}
                                >
                                    pH
                                </div>
                                <div
                                    className={`select-box ${this.state.showDataTemperature ? "active" : ""}`}
                                    onClick={() => this.handleShowAnalysis(2)}
                                >
                                    Temperature
                                </div>
                                <div
                                    className={`select-box ${this.state.showDataTurbidity ? "active" : ""}`}
                                    onClick={() => this.handleShowAnalysis(3)}
                                >
                                    Turbidity
                                </div>
                                <div
                                    className={`select-box ${this.state.showDatatds ? "active" : ""}`}
                                    onClick={() => this.handleShowAnalysis(4)}
                                >
                                    TDS
                                </div>
                            </div>


                            <BasicLineChart
                                xdata={this.state.sensorDataTime}
                                ydata={
                                    this.state.showDataPh ? this.state.sensorDataPh :
                                        this.state.showDataTemperature ? this.state.sensorDataTemperature :
                                            this.state.showDataTurbidity ? this.state.sensorDataTurbidity :
                                                this.state.showDatatds ? this.state.sensorDatatds :
                                                    []
                                }
                            />

                        </div>
                    </div>
                </div>
            </div>
        );
    }
}

export default ShowImages;
