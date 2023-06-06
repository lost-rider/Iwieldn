/* eslint-disable jsx-a11y/anchor-is-valid */
import "../App.css";
import React from "react";
import { Link } from "react-router-dom";
import Sidebar from "./Sidebar";
import Navbar from "./Navbar";
const CoeamtPage = () => {
  return (
    <div className="rowC">
      <Sidebar
        isGUI="false"
        isCoeamt="true"
        isUD="false"
        isRR="false"
        isContact="false"
      />
      <Navbar isGUI="false" isCoeamt="true" isUD="false" isRR="false" isContact="false"/>
      <div className="Page">
        <div className="pagecontainer">
          <img src={require("../images/coeamt2.jpg")} alt="info" />
          <p>
            The Centre of Excellence in Advanced Manufacturing Technology has
            been established at IIT Kharagpur through the support of the
            Department of Heavy Industry of Ministry of Heavy Industries and
            Public Enterprises, Government of India, along with a consortium of
            top industry members in the country. The centre aims to stimulate
            the innovation to manufacture smart machines in the capital goods
            sector. The center will bring together various industries in this
            area to work in a synergistic way towards the common goals of
            infusing cutting edge technologies, and to come up with research and
            development for sustainable products having higher productivity with
            reduced cost.
          </p>

          <p>
            This centre offers a unique platform for collaborative, consortium
            driven infusion of advanced technologies in the manufacturing area,
            which is in harmony with the 'Make-in-India' initiative of the
            Government of India. The centre will initiates innovative and
            top-quality research focused to the industries on Specialty
            materials, Design and automation, Additive manufacturing, and
            Digital manufacturing and Industrial Internet of Things. The centre
            will boost innovative interventions in the advanced manufacturing
            domain by enabling an ecosystem among Institutes of higher repute,
            heavy industries, and also the MSMEs and start-ups. The centre looks
            for active participation in this ecosystem for a collaborative
            research in the proposed areas.
          </p>

          <p>
            The centre also houses an Innovation Lab to facilitate the culture
            of innovation and open engineering. The Innovation Lab invites MSMEs
            and the Start-ups to grab opportunities of getting an end-to-end
            support from the experts including access to various
            state-of-the-art facilities for early prototyping of their product.
            The centre also welcomes bright and talented scholars with high
            value doctoral fellowship to support its activities.
          </p>
          <Link to="/AboutCoeamtPage">
            <a className="btn btn-dark " href="#" role="button">
              Back
            </a>
          </Link>
        </div>
      </div>
    </div>
  );
};

export default CoeamtPage;
