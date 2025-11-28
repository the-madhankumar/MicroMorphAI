import './App.css';
import {
  BrowserRouter as Router,
  Routes,
  Route
} from "react-router-dom";

import Header from "./components/Header";
import Home from './components/Home';
import UserInput from "./components/UserInput";

function App() {
  return (
    <Router>
      <Header />
      <Routes>
        <Route path="/" element={<Home />} />
        <Route exact path="/about" element={<Home />} />
        <Route exact path="/detect" element={<Home />} />
        <Route exact path="/userinput" element={<UserInput/>}/>
      </Routes>
    </Router>
  );
}

export default App;
