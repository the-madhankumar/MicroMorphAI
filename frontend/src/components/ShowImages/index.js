import { Component } from "react";
import SimpleCharts from "./BarChart";
import CompositionExample from "./GaugeChart"
import showImagesList from "../../Data/showImage.js";
import { ArrowBigRightDash, ArrowBigLeftDash } from "lucide-react";

import './index.css';

class ShowImages extends Component {
    state = {
        showImageData: [],
        carouselIndex: 1,
        currentImageSpec: null
    };

    componentDidMount() {
        this.setState({
            showImageData: showImagesList,
            currentImageSpec: showImagesList[1] || null
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

    render() {
        const { showImageData, carouselIndex, currentImageSpec } = this.state;

        const mainImage = showImageData[0];
        const currentImage = showImageData[carouselIndex];

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
        );
    }
}

export default ShowImages;
