/* eslint-disable no-unused-vars */
import "../App.css";
import React from "react";
import { Link } from "react-router-dom";

const Navbar = (props) => {
  return (
    <div>
      <nav className="navbar navbar-expand-lg bg-dark" data-bs-theme="dark">
        <div className="container-fluid">
          <a className="navbar-brand" href="/">
            iWeld
          </a>
          <button
            className="navbar-toggler"
            type="button"
            data-bs-toggle="collapse"
            data-bs-target="#navbarSupportedContent"
            aria-controls="navbarSupportedContent"
            aria-expanded="false"
            aria-label="Toggle navigation"
          >
            <span className="navbar-toggler-icon"></span>
          </button>
          <div className="collapse navbar-collapse" id="navbarSupportedContent">
            <ul className="navbar-nav me-auto mb-2 mb-lg-0">
              <li className={`nav-item ${props.isGUI}`}>
                <Link to="/">
                  <a className="nav-link active" aria-current="page" href="/">
                    About the GUI
                  </a>
                </Link>
              </li>

              <li className={`nav-item ${props.isCoeamt}`}>
                <Link to="/AboutCoeamtPage">
                  <a className="nav-link active" href="/">
                    About Coeamt
                  </a>
                </Link>
              </li>

              <li className={`nav-item ${props.isUD}`}>
                <Link to="/UploadandDetectionPage">
                  <a className="nav-link active" href="/">
                    Uploading & Detection
                  </a>
                </Link>
              </li>
              <li className={`nav-item ${props.isRR}`}>
                <Link to="/UploadandDetectionPage">
                  <a className="nav-link active" href="/">
                    Results & Report Generation
                  </a>
                </Link>
              </li>
              <li className={`nav-item ${props.isContact}`}>
                <Link to="/Contact">
                  <a className="nav-link active" href="/">
                    Contact Us
                  </a>
                </Link>
              </li>
            </ul>
          </div>
        </div>
      </nav>
    </div>
  );
};

export default Navbar;
