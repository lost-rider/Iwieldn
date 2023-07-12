/* eslint-disable jsx-a11y/anchor-is-valid */
import "../App.css";
import React, { useState } from "react";
import Sidebar from "./Sidebar";
import Navbar from "./Navbar";
// import axios from "axios";
import "../index.tsx";
import { ProgressBar } from 'react-bootstrap';
// import { Link } from "react-router-dom";


// let files = [...e.target.files.slice(0, 4)]
import axios from "axios";


const UploadandDetectionPage = () => {
  // const [uploadPercentage, setUploadPercentage] = useState(0);

  // const handleFileUpload = async (e) => {
  //   const files = Array.from(e.target.files);
  //   const totalFiles = files.length;

  //   for (let i = 0; i < totalFiles; i++) {
  //     const file = files[i];
  //     const formData = new FormData();
  //     formData.append('file', file);

  //     try {
  //       const response = await axios.post('/upload', formData, {
  //         onUploadProgress: (progressEvent) => {
  //           const percentage = Math.round(
  //             (progressEvent.loaded * 100) / progressEvent.total
  //           );

  //           setUploadPercentage(percentage);
  //         },
  //       });

  //       console.log('Upload response:', response.data);
  //     } catch (error) {
  //       console.error('Upload error:', error);
  //     }
  //   }

  //   setUploadPercentage(0);
  // };



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
      <Navbar isGUI="false" isCoeamt="false" isUD="true" isRR="false" isContact="false" />
      <div className="Page2">
        <div className="uploadimage">
          <div className="text-logo">
            <div className="text">
              <h6>Step 1(a) : Upload image in DICOM format</h6>
              <p>(Maximum 30 at a time)</p>
            </div>
            <img src={require("../images/upload-logo.png")} alt="info" />
          </div>
          <p>Uploading files :</p>

          <div className="box">
            <input type="file" multiple accept="*/dicom,.dcm, image/dcm, */dcm, .dicom" onChange={handleUpload} />
          </div>

          <p>Uploading status :</p>
        </div>
        <div className="detection">
          <div className="text-logo">
            <div className="text">
              <h6>Step 1(b) : Detection</h6>
            </div>
            <img src={require("../images/search.png")} alt="info" />
          </div>
          <div className="box">

          </div>
          <p>Detection status :</p>
          <div class="container">
            <h1>Enter the welding parameters</h1>
            <form action="noaction.php">
              <div class="formgrp">
                <input type="text" name="" placeholder="Length of object"></input>
              </div>

              <div class="formgrp">
                <input type="text" name="" placeholder="Position of joint"></input>
              </div>

              <div class="formgrp">
                <input type="text" name="" placeholder="Shape of object"></input>
              </div>



              <button class="btn">Submit</button>
            </form>
          </div>
        </div>
      </div>
    </div>



  );
};

export default UploadandDetectionPage;
