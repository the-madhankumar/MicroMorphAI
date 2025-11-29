import React from "react";
import { FileInput } from 'lucide-react';
import { withRouter } from '../../withRouter';

import home_data from "../../Data/home";
import DataCard from "../DataCard/index";
import './index.css';

class Home extends React.Component {
    state = {
        homeData: []
    }

    componentDidMount() {
        this.setState({
            homeData: home_data.steps
        });
    }

    handleUserInputButton = () => {
        this.props.navigate('/userinput');
    }

    render() {
        const { homeData } = this.state;

        return (
            <div className="Home-container">
                <div className="first-page">
                    <img src="/Images/logofirst.png" alt="logo" width="400" height="400" />
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
                        The AI System that detects<br />what the eye cannot see
                    </h1>

                    <button type="button" className="custom-button" onClick={this.handleUserInputButton}>
                        <span className="btn-text">User Input</span>
                        <span className="btn-file"><FileInput /></span>
                    </button>
                </div>

                <ul className="instructions-container">
                    {homeData.map((item) => (
                        <li className="each-one-container" key={item.step}>
                            <DataCard
                                title={item.title}
                                description={item.description}
                            />
                        </li>
                    ))}
                </ul>

                <div className="footer">
                    <span className="footer-left">© 2025 LyMora</span>
                    <span className="footer-right">Deep Vision</span>
                </div>
            </div>
        );
    }
}

export default withRouter(Home);
