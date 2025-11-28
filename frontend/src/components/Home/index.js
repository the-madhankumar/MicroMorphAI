import React from "react";
import home_data from "../../Data/home";
import DataCard from "../DataCard/index";
import logo from "../../Images/logofirst.png";
import './index.css'

class Home extends React.Component {
    state = {
        homeData: []
    }

    componentDidMount() {
        this.setState({
            homeData: home_data.steps
        })
    }

    render() {
        const { homeData } = this.state
        return (
            <div className="Home-container">
                <div className="first-page">
                    <img src={logo} alt="logo" width="500" height="500" />
                    <div className="global-description">
                        <ul>
                            <li>This app automatically detects, classifies, and counts microscopic marine organisms using AI.</li>
                            <li>It combines five intelligent models to achieve highly accurate species identification.</li>
                            <li>The system also tracks important water-quality parameters such as pH, turbidity, temperature, and GPS.</li>
                            <li>It can identify new or unseen organisms using open-set recognition.</li>
                            <li>The app continuously improves itself through automated model retraining.</li>
                        </ul>
                    </div>
                </div>

                <div className="next-button">
                    <h1 className="caption">
                        The System that finds what you cannot see
                    </h1>
                    <button type="button" className="custom-button">
                        UserInput
                    </button>
                </div>

                <ul className="instructions-container">
                    {
                        homeData.map((item) => (
                            <li className="each-one-container" key={item.step}>
                                <DataCard
                                    title={item.title}
                                    description={item.description}
                                />
                            </li>
                        ))
                    }
                </ul>

            </div>
        )
    }
}

export default Home