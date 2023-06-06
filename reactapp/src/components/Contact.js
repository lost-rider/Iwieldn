/* eslint-disable jsx-a11y/anchor-is-valid */
import "../App.css";
import React from "react";
import Sidebar from "./Sidebar";
import Navbar from "./Navbar";
const Contact = () => {
  return (
    <div className="rowC">
      <Sidebar
        isGUI="false"
        isCoeamt="false"
        isUD="false"
        isRR="false"
        isContact="true"
      />
      <Navbar
        isGUI="false"
        isCoeamt="false"
        isUD="false"
        isRR="false"
        isContact="true"
      />
      <div className="Page">
        <div className="contactinfo">
          <form>
            <div className="name">
              <label htmlFor="exampleInputEmail1" className="form-label">
                Name
              </label>
              <input
                type="email"
                className="form-control"
                id="exampleInputEmail1"
                aria-describedby="emailHelp"
              />
            </div>
            <div className="email">
              <label htmlFor="exampleInputEmail1" className="form-label">
                Email address
              </label>
              <input
                type="email"
                className="form-control"
                id="exampleInputEmail1"
                aria-describedby="emailHelp"
              />
              <div id="emailHelp" className="form-text">
                We'll never share your email and phone with anyone else.
              </div>
            </div>
            <div className="phone">
              <label htmlFor="clientphone" className="form-label">
                PhoneNumber
              </label>
              <input
                type="password"
                className="form-control"
                id="clientphone"
              />
            </div>
            <div className="check">
              <input
                type="checkbox"
                className="form-check-input"
                id="exampleCheck1"
              />
              <label className="form-check-label" htmlFor="exampleCheck1">
                Check me out
              </label>
            </div>
            <div className="enquiry">
              <label htmlFor="enquiry" className="form-label">
                Enquiry
              </label>
              <textarea
                className="form-control"
                aria-label="With textarea"
              ></textarea>
            </div>
            <div className="buttons">
              <button type="submit">Submit</button>
            </div>
          </form>
        </div>
      </div>
    </div>
  );
};

export default Contact;
