import "../App.css";
import React from "react";
import { Link } from "react-router-dom";

const Sidebar = (props) => {
  return (
    <div>
      <div className="sidebar">
        <div className="sidebarhead">
          <h4>DOCUMENT</h4>
        </div>
        <div className="sidebarcontainer">
          <div className="image">
            <img src={require("../images/info-icon.png")} alt="info" />
          </div>
          <div className="text">
            <Link to="/">
              <a className={`${props.isGUI}`} href="#about">About The GUI</a>
            </Link>
          </div>
        </div>
        <div className="sidebarcontainer">
          <div className="image">
            <img src={require("../images/info-icon.png")} alt="info" />
          </div>
          <div className="text">
            <Link to="/AboutCoeamtPage">
              <a className={`${props.isCoeamt}`}href="#about">About Coeamt</a>
            </Link>
          </div>
        </div>
        <hr />
        <div className="sidebarhead">
          <h4>ANALYSIS</h4>
        </div>
        <div className="sidebarcontainer">
          <div className="image">
            <img src={require("../images/logo1.png")} alt="info" />
          </div>
          <div className="text">
            <Link to="/UploadandDetectionPage">
              <a className={`${props.isUD}`} href="#about">Uploading & Detection</a>
            </Link>
          </div>
        </div>
        <div className="sidebarcontainer">
          <div className="image">
            <img src={require("../images/logo2.png")} alt="info" />
          </div>
          <div className="text">
            <a className={`${props.isRR}`} href="#about">Results & Report Generation</a>
          </div>
        </div>
        <hr />
        <div className="sidebarhead">
          <h4>CONTACT</h4>
        </div>
        <div className="sidebarcontainer">
          <div className="image">
            <img src={require("../images/contactus.png")} alt="info" />
          </div>
          <div className="text">
            <Link to="/Contact">
              <a className={`${props.isContact}`} href="#about">Contact Us</a>
            </Link>
          </div>
        </div>
        <hr />
      </div>
    </div>
  );
};

export default Sidebar;
