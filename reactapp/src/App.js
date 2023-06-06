/* eslint-disable react/jsx-no-undef */
/* eslint-disable no-unused-vars */
import "./App.css";
import Sidebar from "./components/Sidebar";
import AboutiWeldPage from "./components/AboutiWeldPage";
import AboutCoeamtPage from "./components/AboutCoeamtPage";
import CoeamtPage from "./components/CoeamtPage";
import Contact from "./components/Contact";
import DownloadManual from "./components/DownloadManual";
import UploadandDetectionPage from "./components/UploadandDetectionPage";
import Navbar from "./components/Navbar";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";

function App() {
  return (
    <>
      <Router>
        <div >
          {/* <Navbar /> */}
          {/* <Sidebar /> */}
          <Routes>
            <Route exact path="" element={<AboutiWeldPage />}></Route>
            <Route
              exact
              path="/AboutCoeamtPage"
              element={<AboutCoeamtPage />}
            ></Route>
            <Route exact path="/CoeamtPage" element={<CoeamtPage />}></Route>
            <Route exact path="/Contact" element={<Contact />}></Route>
            <Route
              exact
              path="/UploadandDetectionPage" element={<UploadandDetectionPage/>}
            ></Route>
            <Route
              exact
              path="/DownloadManual" element={<DownloadManual />}
            ></Route>
          </Routes>
        </div>
      </Router>
    </>
  );
}

export default App;
