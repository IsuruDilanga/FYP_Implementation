

//////// vertion 3

import React, { useEffect, useState, useRef } from "react";
import "./Home.css";
import { Link } from "react-router-dom";

const HomePage = () => {

    const [mediaStream, setMediaStream] = useState(null);
    const [capturedImage, setCapturedImage] = useState(null);
    const [selectedFile, setSelectedFile] = useState(null);
    const [uploadedImage, setUploadedImage] = useState(null);
    const [showUploadedImage, setShowUploadedImage] = useState(false);
    const videoRef = useRef(null);
    const fileInputRef = useRef(null);
    const [showUploadButton, setShowUploadButton] = useState(true);
    const [imagePath, setImagePath] = useState(null);

    let responseData;

    const startCamera = async () => {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ video: true });
            setMediaStream(stream);
            if (videoRef.current) {
                videoRef.current.srcObject = stream;
                videoRef.current.play();
                videoRef.current.width = 400;
                videoRef.current.height = 400;
            }
        } catch (error) {
            console.error('Error accessing camera:', error);
        }
    };

    const stopCamera = () => {
        mediaStream.getTracks().forEach(track => {
            track.stop();
        });
        setMediaStream(null);
    };

    const restartCamera = () => {
        if (videoRef.current && mediaStream) {
            videoRef.current.srcObject = mediaStream;
            videoRef.current.play();
        }
        setCapturedImage(null);
    };

    const captureImage = async () => {
        if (videoRef.current) {
            const canvas = document.createElement('canvas');
            canvas.width = videoRef.current.videoWidth;
            canvas.height = videoRef.current.videoHeight;
            const ctx = canvas.getContext('2d');
            ctx.drawImage(videoRef.current, 0, 0);

            try {
                const timestamp = Date.now();
                const fileName = `captured_image_${timestamp}.jpg`;

                const blob = await new Promise((resolve, reject) => {
                    canvas.toBlob(blob => {
                        if (!blob) {
                            reject('Error creating blob');
                        }
                        resolve(blob);
                    }, 'image/jpeg');
                });

                const formData = new FormData();
                formData.append('image', blob, fileName);

                const response = await fetch('http://localhost:8000/save_image', {
                    method: 'POST',
                    body: formData,
                });

                if (response.ok) {
                    responseData = await response.json();
                    setCapturedImage(responseData.filename);
                    console.log('Image saved successfully. Filename:', responseData.filename);
                } else {
                    alert('Failed to save image.');
                }
            } catch (error) {
                console.error('Error saving image:', error);
                alert('Failed to save image.');
            } finally {
                if (videoRef.current) {
                    videoRef.current.pause();
                }
            }
        }
    };

    const handleFileChange = (event) => {
        setShowUploadedImage(false); // Hide uploaded image
        setUploadedImage(null); // Reset uploaded image
        const file = event.target.files[0];
        if (file && file.type === 'image/jpeg') {
            setSelectedFile(file);
            uploadImage(file);
        } else {
            setSelectedFile(null);
            alert('Please select a JPG image.');
        }
    };

    const uploadImage = async (file) => {
        try {
            const formData = new FormData();
            formData.append('image', file);

            const response = await fetch('http://localhost:8000/upload_image', {
                method: 'POST',
                body: formData,
            });

            if (response.ok) {
                console.log('Image uploaded successfully.');
                responseData = await response.json();
                console.log('Image saved successfully. Filename:', responseData.filename);

                setImagePath(responseData.filename);

                setUploadedImage(URL.createObjectURL(file));
                setMediaStream(null);
                setShowUploadedImage(true); // Show uploaded image

            } else {
                alert('Failed to upload image.');
            }
        } catch (error) {
            console.error('Error uploading image:', error);
            alert('Failed to upload image.');
        }
    };

    const handleBackButtonClick = () => {
        setShowUploadedImage(false);
        setShowUploadButton(true); // Show upload button when back button is clicked
    };

    const predictASDchild = async () => {
        console.log("hello ", imagePath);;
        try {
            const response = await fetch(`http://localhost:8000/predictASD?filepath=${imagePath}`);
            if (response.ok) {
                const data = await response.json();
                console.log('Prediction successful:', data);
                // Handle response data
            } else {
                console.error('Failed to predict ASD:', response.statusText);
                // Handle error
            }
        } catch (error) {
            console.error('Failed to predict ASD:', error);
            // Handle error
        }
    }

    return (
        <div>
            <nav className="navbar">
                <div className="container">
                    <a className="navbar-brand">Navbar</a>
                    <form className="d-flex" role="search">
                        <input className="form-control me-2" type="search" placeholder="Search" aria-label="Search" />
                        <button className="btn btn-outline-success" type="submit">Search</button>
                    </form>
                </div>
            </nav>
            <div className="container">
                <div className="row">
                    <div className="col-lg-8" style={{ backgroundColor: 'red' }}>
                        <div className="heading">
                            <div className="title">
                                <p>Empowering <br></br> Understanding</p>
                            </div>
                            <div className="sub-title">
                                <p>ASD and Emotion detection with <br></br> Explainable AI insights</p>
                            </div>
                        </div>
                    </div>
                    <div className="col-lg-4">
                        <div className="video-container">
                            <video ref={videoRef} autoPlay muted className="video" style={{ display: mediaStream ? 'block' : 'none' }} />
                            {showUploadedImage && (
                                <div>
                                    <img src={uploadedImage} alt="Uploaded" className="uploaded-image" /><br></br>
                                    <button onClick={handleBackButtonClick} className="btn btn-primary btn-back">Back</button>
                                </div>
                            )}
                            {!showUploadedImage && !mediaStream && (
                                <button onClick={startCamera} className="btn btn-info btn-camera">
                                    Open Camera
                                </button>
                            )}
                            {mediaStream && !capturedImage && (
                                <button onClick={captureImage} className="btn btn-primary btn-capture">Capture Image</button>
                            )}
                            {mediaStream && !capturedImage && (
                                <button onClick={stopCamera} className="btn btn-danger">Close Camera</button>
                            )}
                            {capturedImage && (
                                <button onClick={restartCamera} className="btn btn-primary btn-restart">
                                    Restart Camera
                                </button>
                            )}
                            {!mediaStream && !showUploadedImage && !capturedImage && (
                                <div>
                                    <input
                                        type="file"
                                        accept="image/jpeg"
                                        onChange={handleFileChange}
                                        ref={fileInputRef}
                                        style={{ display: 'none' }}
                                    />
                                    <button onClick={() => fileInputRef.current.click()} className="btn btn-primary btn-upload">Upload Image</button>
                                </div>
                            )}
                            {(showUploadedImage || capturedImage) && (
                                <div>
                                    <button onClick={predictASDchild} className="btn btn-primary btn-upload">Check ASD</button>
                                </div>
                            )}
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default HomePage;


/////

// import React, { useEffect, useState, useRef } from "react";
// import "./Home.css";
// import { Link } from "react-router-dom";

// const HomePage = () => {

//     const [mediaStream, setMediaStream] = useState(null);
//     const [capturedImage, setCapturedImage] = useState(null);
//     const [selectedFile, setSelectedFile] = useState(null);
//     const [uploadedImage, setUploadedImage] = useState(null);
//     const [showUploadedImage, setShowUploadedImage] = useState(false);
//     const [showUploadButton, setShowUploadButton] = useState(true); // New state for showing/hiding the upload button
//     const videoRef = useRef(null);
//     const fileInputRef = useRef(null);

//     const startCamera = async () => {
//         try {
//             const stream = await navigator.mediaDevices.getUserMedia({ video: true });
//             setMediaStream(stream);
//             if (videoRef.current) {
//                 videoRef.current.srcObject = stream;
//                 videoRef.current.play();
//                 videoRef.current.width = 400;
//                 videoRef.current.height = 400;
//             }
//         } catch (error) {
//             console.error('Error accessing camera:', error);
//         }
//     };

//     const stopCamera = () => {
//         mediaStream.getTracks().forEach(track => {
//             track.stop();
//         });
//         setMediaStream(null);
//     };

//     const restartCamera = () => {
//         if (videoRef.current && mediaStream) {
//             videoRef.current.srcObject = mediaStream;
//             videoRef.current.play();
//         }
//         setCapturedImage(null);
//     };

//     const captureImage = async () => {
//         if (videoRef.current) {
//             const canvas = document.createElement('canvas');
//             canvas.width = videoRef.current.videoWidth;
//             canvas.height = videoRef.current.videoHeight;
//             const ctx = canvas.getContext('2d');
//             ctx.drawImage(videoRef.current, 0, 0);

//             try {
//                 const timestamp = Date.now();
//                 const fileName = `captured_image_${timestamp}.jpg`;

//                 const blob = await new Promise((resolve, reject) => {
//                     canvas.toBlob(blob => {
//                         if (!blob) {
//                             reject('Error creating blob');
//                         }
//                         resolve(blob);
//                     }, 'image/jpeg');
//                 });

//                 const formData = new FormData();
//                 formData.append('image', blob, fileName);

//                 const response = await fetch('http://localhost:8000/save_image', {
//                     method: 'POST',
//                     body: formData,
//                 });

//                 if (response.ok) {
//                     const responseData = await response.json();
//                     setCapturedImage(responseData.filename);
//                     console.log('Image saved successfully. Filename:', responseData.filename);
//                     setShowUploadButton(false); // Hide upload button after capturing image
//                 } else {
//                     alert('Failed to save image.');
//                 }
//             } catch (error) {
//                 console.error('Error saving image:', error);
//                 alert('Failed to save image.');
//             } finally {
//                 if (videoRef.current) {
//                     videoRef.current.pause();
//                 }
//             }
//         }
//     };

//     const handleFileChange = (event) => {
//         setShowUploadedImage(false); // Hide uploaded image
//         setUploadedImage(null); // Reset uploaded image
//         const file = event.target.files[0];
//         if (file && file.type === 'image/jpeg') {
//             setSelectedFile(file);
//             uploadImage(file);
//         } else {
//             setSelectedFile(null);
//             alert('Please select a JPG image.');
//         }
//     };

//     const uploadImage = async (file) => {
//         try {
//             const formData = new FormData();
//             formData.append('image', file);

//             const response = await fetch('http://localhost:8000/upload_image', {
//                 method: 'POST',
//                 body: formData,
//             });

//             if (response.ok) {
//                 console.log('Image uploaded successfully.');
//                 setUploadedImage(URL.createObjectURL(file));
//                 setMediaStream(null);
//                 setShowUploadedImage(true); // Show uploaded image
//                 setShowUploadButton(false); // Hide upload button after uploading image
//             } else {
//                 alert('Failed to upload image.');
//             }
//         } catch (error) {
//             console.error('Error uploading image:', error);
//             alert('Failed to upload image.');
//         }
//     };

//     const handleBackButtonClick = () => {
//         setShowUploadedImage(false);
//         setShowUploadButton(true); // Show upload button when back button is clicked
//     };

//     return (
//         <div>
//             <nav className="navbar">
//                 <div className="container">
//                     <a className="navbar-brand">Navbar</a>
//                     <form className="d-flex" role="search">
//                         <input className="form-control me-2" type="search" placeholder="Search" aria-label="Search" />
//                         <button className="btn btn-outline-success" type="submit">Search</button>
//                     </form>
//                 </div>
//             </nav>
//             <div className="container">
//                 <div className="row">
//                     <div className="col-lg-8" style={{ backgroundColor: 'red' }}>
//                         <div className="heading">
//                             <div className="title">
//                                 <p>Empowering <br></br> Understanding</p>
//                             </div>
//                             <div className="sub-title">
//                                 <p>ASD and Emotion detection with <br></br> Explainable AI insights</p>
//                             </div>
//                         </div>
//                     </div>
//                     <div className="col-lg-4">
//                         <div className="video-container">
//                             <video ref={videoRef} autoPlay muted className="video" style={{ display: mediaStream ? 'block' : 'none' }} />
//                             {showUploadedImage && (
//                                 <div>
//                                     <img src={uploadedImage} alt="Uploaded" className="uploaded-image" /><br></br>
//                                     <button onClick={handleBackButtonClick} className="btn btn-primary btn-back">Back</button>
//                                 </div>
//                             )}
//                             {!showUploadedImage && !mediaStream && !capturedImage && showUploadButton && (
//                                 <button onClick={startCamera} className="btn btn-info btn-camera">
//                                     Open Camera
//                                 </button>
//                             )}
//                             {mediaStream && !capturedImage && (
//                                 <button onClick={captureImage} className="btn btn-primary btn-capture">Capture Image</button>
//                             )}
//                             {mediaStream && !capturedImage && (
//                                 <button onClick={stopCamera} className="btn btn-danger">Close Camera</button>
//                             )}
//                             {capturedImage && (
//                                 <button onClick={restartCamera} className="btn btn-primary btn-restart">
//                                     Restart Camera
//                                 </button>
//                             )}
//                             {!mediaStream && !showUploadedImage && !capturedImage && (
//                                 <div>
//                                     <input
//                                         type="file"
//                                         accept="image/jpeg"
//                                         onChange={handleFileChange}
//                                         ref={fileInputRef}
//                                         style={{ display: 'none' }}
//                                     />
//                                     <button onClick={() => fileInputRef.current.click()} className="btn btn-primary btn-upload">Upload Image</button>
//                                 </div>
//                             )}
//                         </div>
//                     </div>
//                 </div>
//             </div>
//         </div>
//     );
// };

// export default HomePage;


////////  VErsion 2

// import React, { useEffect, useState, useRef } from "react";
// import "./Home.css";
// import { Link } from "react-router-dom";

// const HomePage = () => {

//     const [mediaStream, setMediaStream] = useState(null);
//     const [capturedImage, setCapturedImage] = useState(null);
//     const [selectedFile, setSelectedFile] = useState(null);
//     const [uploadedImage, setUploadedImage] = useState(null);
//     const videoRef = useRef(null);
//     const fileInputRef = useRef(null);

//     const startCamera = async () => {
//         try {
//             const stream = await navigator.mediaDevices.getUserMedia({ video: true });
//             setMediaStream(stream);
//             if (videoRef.current) {
//                 videoRef.current.srcObject = stream;
//                 videoRef.current.play();
//                 videoRef.current.width = 400;
//                 videoRef.current.height = 400;
//             }
//         } catch (error) {
//             console.error('Error accessing camera:', error);
//         }
//     };

//     const stopCamera = () => {
//         mediaStream.getTracks().forEach(track => {
//             track.stop();
//         });
//         setMediaStream(null);
//     };

//     const restartCamera = () => {
//         if (videoRef.current && mediaStream) {
//             videoRef.current.srcObject = mediaStream;
//             videoRef.current.play();
//         }
//         setCapturedImage(null);
//     };

//     const captureImage = async () => {
//         if (videoRef.current) {
//             const canvas = document.createElement('canvas');
//             canvas.width = videoRef.current.videoWidth;
//             canvas.height = videoRef.current.videoHeight;
//             const ctx = canvas.getContext('2d');
//             ctx.drawImage(videoRef.current, 0, 0);

//             try {
//                 const timestamp = Date.now();
//                 const fileName = `captured_image_${timestamp}.jpg`;

//                 const blob = await new Promise((resolve, reject) => {
//                     canvas.toBlob(blob => {
//                         if (!blob) {
//                             reject('Error creating blob');
//                         }
//                         resolve(blob);
//                     }, 'image/jpeg');
//                 });

//                 const formData = new FormData();
//                 formData.append('image', blob, fileName);

//                 const response = await fetch('http://localhost:8000/save_image', {
//                     method: 'POST',
//                     body: formData,
//                 });

//                 if (response.ok) {
//                     const responseData = await response.json();
//                     setCapturedImage(responseData.filename);
//                     console.log('Image saved successfully. Filename:', responseData.filename);
//                 } else {
//                     alert('Failed to save image.');
//                 }
//             } catch (error) {
//                 console.error('Error saving image:', error);
//                 alert('Failed to save image.');
//             } finally {
//                 if (videoRef.current) {
//                     videoRef.current.pause();
//                 }
//             }
//         }
//     };

//     const handleFileChange = (event) => {
//         const file = event.target.files[0];
//         if (file && file.type === 'image/jpeg') {
//             setSelectedFile(file);
//             uploadImage(file);
//         } else {
//             setSelectedFile(null);
//             alert('Please select a JPG image.');
//         }
//     };

//     const uploadImage = async (file) => {
//         try {
//             const formData = new FormData();
//             formData.append('image', file);

//             const response = await fetch('http://localhost:8000/upload_image', {
//                 method: 'POST',
//                 body: formData,
//             });

//             if (response.ok) {
//                 console.log('Image uploaded successfully.');
//                 setUploadedImage(URL.createObjectURL(file));
//                 setMediaStream(null);
//             } else {
//                 alert('Failed to upload image.');
//             }
//         } catch (error) {
//             console.error('Error uploading image:', error);
//             alert('Failed to upload image.');
//         }
//     };

//     return (
//         <div>
//             <nav className="navbar">
//                 <div className="container">
//                     <a className="navbar-brand">Navbar</a>
//                     <form className="d-flex" role="search">
//                         <input className="form-control me-2" type="search" placeholder="Search" aria-label="Search" />
//                         <button className="btn btn-outline-success" type="submit">Search</button>
//                     </form>
//                 </div>
//             </nav>
//             <div className="container">
//                 <div className="row">
//                     <div className="col-lg-8" style={{ backgroundColor: 'red' }}>
//                         <div className="heading">
//                             <div className="title">
//                                 <p>Empowering <br></br> Understanding</p>
//                             </div>
//                             <div className="sub-title">
//                                 <p>ASD and Emotion detection with <br></br> Explainable AI insights</p>
//                             </div>
//                         </div>
//                     </div>
//                     <div className="col-lg-4">
//                         <div className="video-container">
//                             <video ref={videoRef} autoPlay muted className="video" style={{ display: mediaStream ? 'block' : 'none' }} />
//                             {uploadedImage && (
//                                 <img src={uploadedImage} alt="Uploaded" className="uploaded-image" />
//                             )}
//                             {!uploadedImage && !mediaStream && (
//                                 <button onClick={startCamera} className="btn btn-info btn-camera">
//                                     Open Camera
//                                 </button>
//                             )}
//                             {mediaStream && !capturedImage && (
//                                 <button onClick={captureImage} className="btn btn-primary btn-capture">Capture Image</button>
//                             )}
//                             {mediaStream && !capturedImage && (
//                                 <button onClick={stopCamera} className="btn btn-danger">Close Camera</button>
//                             )}
//                             {capturedImage && (
//                                 <button onClick={restartCamera} className="btn btn-primary btn-restart">
//                                     Restart Camera
//                                 </button>
//                             )}
//                             {!mediaStream && (
//                                 <div>
//                                     <input
//                                         type="file"
//                                         accept="image/jpeg"
//                                         onChange={handleFileChange}
//                                         ref={fileInputRef}
//                                         style={{ display: 'none' }}
//                                     />
//                                     <button onClick={() => fileInputRef.current.click()} className="btn btn-primary btn-upload">Upload Image</button>
//                                 </div>
//                             )}
//                         </div>
//                     </div>
//                 </div>
//             </div>
//         </div>
//     );
// };

// export default HomePage;



/////// Version 1

// import React, { useEffect, useState, useRef } from "react";
// import "./Home.css";
// import { Link } from "react-router-dom";

// const HomePage = () => {

//     const [mediaStream, setMediaStream] = useState(null);
//     const [capturedImage, setCapturedImage] = useState(null);
//     const [selectedFile, setSelectedFile] = useState(null);
//     const videoRef = useRef(null);
//     const fileInputRef = useRef(null);

//     const startCamera = async () => {
//         try {
//             const stream = await navigator.mediaDevices.getUserMedia({ video: true });
//             setMediaStream(stream);
//             if (videoRef.current) {
//                 videoRef.current.srcObject = stream;
//                 videoRef.current.play(); // Ensure video is playing
//                 videoRef.current.width = 400; // Set width to 400
//                 videoRef.current.height = 400; // Set height to 400
//             }
//         } catch (error) {
//             console.error('Error accessing camera:', error);
//         }
//     };

//     const stopCamera = () => {
//         // if (mediaStream) {
//         //     mediaStream.getTracks().forEach(track => {
//         //         track.stop();
//         //     });
//         //     setMediaStream(null);
//         // }
//         mediaStream.getTracks().forEach(track => {
//             track.stop();
//         });
//         setMediaStream(null);
//     };

//     const restartCamera = () => {
//         if (videoRef.current && mediaStream) {
//             videoRef.current.srcObject = mediaStream;
//             videoRef.current.play();
//         }
//         setCapturedImage(null); // Clear captured image
//     };

//     const captureImage = async () => {
//         if (videoRef.current) {
//             const canvas = document.createElement('canvas');
//             canvas.width = videoRef.current.videoWidth;
//             canvas.height = videoRef.current.videoHeight;
//             const ctx = canvas.getContext('2d');
//             ctx.drawImage(videoRef.current, 0, 0);

//             try {
//                 // Generate a unique file name using timestamp
//                 const timestamp = Date.now();
//                 const fileName = `captured_image_${timestamp}.jpg`;

//                 // Convert canvas data to a blob
//                 const blob = await new Promise((resolve, reject) => {
//                     canvas.toBlob(blob => {
//                         if (!blob) {
//                             reject('Error creating blob');
//                         }
//                         resolve(blob);
//                     }, 'image/jpeg');
//                 });

//                 // Create FormData object and append the blob
//                 const formData = new FormData();
//                 formData.append('image', blob, fileName);

//                 const response = await fetch('http://localhost:8000/save_image', {
//                     method: 'POST',
//                     body: formData,
//                 });

//                 if (response.ok) {
//                     const responseData = await response.json();
//                     setCapturedImage(responseData.filename); // Set captured image URL
//                     console.log('Image saved successfully. Filename:', responseData.filename);
//                 } else {
//                     alert('Failed to save image.');
//                 }
//             } catch (error) {
//                 console.error('Error saving image:', error);
//                 alert('Failed to save image.');
//             } finally {
//                 // Pause the camera after capturing the image
//                 if (videoRef.current) {
//                     videoRef.current.pause();
//                 }
//             }
//         }
//     };

//     const handleFileChange = (event) => {
//         const file = event.target.files[0];
//         if (file && file.type === 'image/jpeg') {
//             setSelectedFile(file);
//             uploadImage(file);
//         } else {
//             setSelectedFile(null);
//             alert('Please select a JPG image.');
//         }
//     };

//     const uploadImage = async (file) => {
//         try {
//             const formData = new FormData();
//             formData.append('image', file);

//             const response = await fetch('http://localhost:8000/upload_image', {
//                 method: 'POST',
//                 body: formData,
//             });

//             if (response.ok) {
//                 // Handle successful upload
//                 console.log('Image uploaded successfully.');
//             } else {
//                 alert('Failed to upload image.');
//             }
//         } catch (error) {
//             console.error('Error uploading image:', error);
//             alert('Failed to upload image.');
//         }
//     };

//     return (

//         <div>
//             <nav className="navbar ">
//                 <div className="container">
//                     <a className="navbar-brand">Navbar</a>
//                     <form className="d-flex" role="search">
//                         <input className="form-control me-2" type="search" placeholder="Search" aria-label="Search" />
//                         <button className="btn btn-outline-success" type="submit">Search</button>
//                     </form>
//                 </div>
//             </nav>
//             <div className="container">
//                 <div className="row">
//                     <div className="col-lg-8" style={{ backgroundColor: 'red' }}>
//                         <div className="heading">
//                             <div className="title">
//                                 <p>Empowering <br></br> Understanding</p>
//                             </div>
//                             <div className="sub-title">
//                                 <p>ASD and Emotion detection with <br></br> Explainable AI insights</p>
//                             </div>
//                         </div>
//                     </div>
//                     <div className="col-lg-4">
//                         <div className="video-container">
//                             <video ref={videoRef} autoPlay muted className="video" style={{ display: mediaStream ? 'block' : 'none' }} />
//                             {!capturedImage && !mediaStream && (
//                                 <button onClick={startCamera} className="btn btn-info btn-camera">
//                                     Open Camera
//                                 </button>
//                             )}
//                             {mediaStream && !capturedImage &&
//                                 <button onClick={captureImage} className="btn btn-primary btn-capture">Capture Image</button>
//                             }

//                             {/* <br></br> */}
//                             {mediaStream && !capturedImage && (
//                                 <button onClick={stopCamera} className="btn btn-danger">Close Camera</button>
//                             )}
//                             {capturedImage && (
//                                 <button onClick={restartCamera} className="btn btn-primary btn-restart">
//                                     Restart Camera
//                                 </button>
//                             )}
//                             {!mediaStream && (
//                                 <div>
//                                     <input
//                                         type="file"
//                                         accept="image/jpeg"
//                                         onChange={handleFileChange}
//                                         ref={fileInputRef}
//                                         style={{ display: 'none' }}
//                                     />
//                                     <button onClick={() => fileInputRef.current.click()} class="btn btn-primary btn-upload">Upload Image</button>
//                                 </div>
//                             )}
//                         </div>
//                     </div>
//                 </div>
//             </div>
//         </div>
//     );
// };

// export default HomePage;