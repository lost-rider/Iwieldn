/* eslint-disable jsx-a11y/anchor-is-valid */
import "../App.css";
import React, { useState } from "react";
import Sidebar from "./Sidebar";
import Navbar from "./Navbar";
// import axios from "axios";
// import { ProgressBar } from "react-bootstrap";
// import { Link } from "react-router-dom";

const UploadandDetectionPage = () => {
  // const [uploadPercentage, setUploadPercentage] = useState(0);
  return (
    <div className="rowC">
      <Sidebar
        isGUI="false"
        isCoeamt="false"
        isUD="true"
        isRR="false"
        isContact="false"
      />
      <Navbar isGUI="false" isCoeamt="false" isUD="true" isRR="false" isContact="false"/>
      <div className="Page2">
        <div className="uploadimage">
          <div className="text-logo">
            <div className="text">

              <p>2(a):Results</p>
            </div>
            <img src={require("../images/upload-logo.png")} alt="info" />
          </div>
          <p>Uploaded image:</p>
          <div className="box">
            {}
          </div>
          <p>Detected image:</p>
          <div className="box">
            {}
          </div>          

        </div>
        <div className="detection">
          <div className="text-logo">
            <div className="text">
              <h6>Step 2(b) Report generation:</h6>
            </div>
            <img src={require("../images/research.png")} alt="info" />
          </div>
          <div className="box"></div>

          <div class="container">
  

</div>
        </div>
      </div>
    </div>


  
  );
};

export default UploadandDetectionPage;
