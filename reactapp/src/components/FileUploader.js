import React, { useState } from "react";
import axios from "axios";

const FileUploader = () => {
  const [selectedFile, setSelectedFile] = useState(null);

  const fileSelectedHandler = event => {
    setSelectedFile(event.target.files[0]);
  };


  const uploadHandler = () => {
    if (this.state && this.state.selectedFile) {
      console.log(this.state.selectedFile);
      const formData = new FormData();
      formData.append(
        'myFile',
        this.state.selectedFile,
        this.state.selectedFile.name
      );
      axios.post('../images', formData);
    }
  };
  
  

  return (
    <div className="box">
      <input
        type="file"
        multiple
        accept="*/dicom,.dcm, image/dcm, */dcm, .dicom"
        onChange={fileSelectedHandler}
      />
      <button onClick={uploadHandler}>Upload!</button>
    </div>
  );
};

export default FileUploader;
