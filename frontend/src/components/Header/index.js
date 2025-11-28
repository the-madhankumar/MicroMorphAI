import { Component } from "react";
import { Link } from "react-router-dom";
import logo from "../../Images/logofirst.png";
import './index.css'

class Header extends Component {
    render() {
        return (
            <div className="nav-header">
                <div>
                    <Link to='/'>
                        <img src={logo} alt="logo" />
                    </Link>
                </div>

                <div>
                    <Link to='/detect'>
                        Detect
                    </Link>
                    <Link to='/about'>
                        About
                    </Link>
                </div>
            </div>
        )
    }
}

export default Header