/* eslint-disable jsx-a11y/anchor-is-valid */
import "../App.css";
import React from "react";
import Sidebar from "./Sidebar";
import Navbar from "./Navbar";
import { Link } from "react-router-dom";

const AboutiWeldPage = () => {
  return (
    <div className="rowC">
      <Sidebar isGUI="true" isCoeamt="false" isUD="false" isRR="false" isContact="false"/>
      <Navbar isGUI="true" isCoeamt="false" isUD="false" isRR="false" isContact="false"/>
      <div className="Page">
        {/* <div><a className="btn btn-dark" href="#" role="button">
        Previous
      </a></div> */}
        <div className="pagecontainer">
          <div className="pagebuttons">
            <a className="btn btn-info" href="#" role="button">
              About iWeld
            </a>
            <Link to="/DownloadManual">
              <a className="btn btn-info" href="#" role="button">
                Download Manual
              </a>
            </Link>
          </div>
          <div className="pageimage"></div>
        </div>
        <div>
          <Link to="/AboutCoeamtPage">
            <a className="btn btn-dark Next" href="#" role="button">
              Next
            </a>
          </Link>
        </div>
      </div>
    </div>
  );
};

export default AboutiWeldPage;
