import React, { useEffect, useState, useRef } from "react";
import "./Home.css";
import { Link } from "react-router-dom";

const HomePage = () => {

    const [mediaStream, setMediaStream] = useState(null);
    const [capturedImage, setCapturedImage] = useState(null);
    const [selectedFile, setSelectedFile] = useState(null);
    const videoRef = useRef(null);
    const fileInputRef = useRef(null);

    const startCamera = async () => {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ video: true });
            setMediaStream(stream);
            if (videoRef.current) {
                videoRef.current.srcObject = stream;
                videoRef.current.play(); // Ensure video is playing
                videoRef.current.width = 400; // Set width to 400
                videoRef.current.height = 400; // Set height to 400
            }
        } catch (error) {
            console.error('Error accessing camera:', error);
        }
    };

    const stopCamera = () => {
        // if (mediaStream) {
        //     mediaStream.getTracks().forEach(track => {
        //         track.stop();
        //     });
        //     setMediaStream(null);
        // }
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
        setCapturedImage(null); // Clear captured image
    };

    const captureImage = async () => {
        if (videoRef.current) {
            const canvas = document.createElement('canvas');
            canvas.width = videoRef.current.videoWidth;
            canvas.height = videoRef.current.videoHeight;
            const ctx = canvas.getContext('2d');
            ctx.drawImage(videoRef.current, 0, 0);

            try {
                // Generate a unique file name using timestamp
                const timestamp = Date.now();
                const fileName = `captured_image_${timestamp}.jpg`;

                // Convert canvas data to a blob
                const blob = await new Promise((resolve, reject) => {
                    canvas.toBlob(blob => {
                        if (!blob) {
                            reject('Error creating blob');
                        }
                        resolve(blob);
                    }, 'image/jpeg');
                });

                // Create FormData object and append the blob
                const formData = new FormData();
                formData.append('image', blob, fileName);

                const response = await fetch('http://localhost:8000/save_image', {
                    method: 'POST',
                    body: formData,
                });

                if (response.ok) {
                    const responseData = await response.json();
                    setCapturedImage(responseData.filename); // Set captured image URL
                    console.log('Image saved successfully. Filename:', responseData.filename);
                } else {
                    alert('Failed to save image.');
                }
            } catch (error) {
                console.error('Error saving image:', error);
                alert('Failed to save image.');
            } finally {
                // Pause the camera after capturing the image
                if (videoRef.current) {
                    videoRef.current.pause();
                }
            }
        }
    };

    const handleFileChange = (event) => {
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
                // Handle successful upload
                console.log('Image uploaded successfully.');
            } else {
                alert('Failed to upload image.');
            }
        } catch (error) {
            console.error('Error uploading image:', error);
            alert('Failed to upload image.');
        }
    };

    return (

        <div>
            <nav className="navbar ">
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
                            {!capturedImage && !mediaStream && <button onClick={startCamera} className="btn-camera">Open Camera</button>}
                            {mediaStream && !capturedImage && <button onClick={captureImage} className="btn-capture">Capture Image</button>}
                            {/* <br></br> */}
                            {mediaStream && !capturedImage && <button onClick={stopCamera} className="btn-close">Close Camera</button>}
                            {capturedImage && <button onClick={restartCamera} className="btn-restart">Restart Camera</button>}
                            {!mediaStream && (
                                <div>
                                    <input
                                        type="file"
                                        accept="image/jpeg"
                                        onChange={handleFileChange}
                                        ref={fileInputRef}
                                        style={{ display: 'none' }}
                                    />
                                    <button onClick={() => fileInputRef.current.click()} className="btn-upload">Upload Image</button>
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