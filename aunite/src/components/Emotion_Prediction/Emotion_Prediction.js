import React, { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import './Emotion_Prediction.css';
import LoadingOverlay from "../LoadingOverlay/LoadingOverlay";
import { toast, ToastContainer } from 'react-toastify';
import 'react-toastify/dist/ReactToastify.css';

const Emotion_Prediction = () => {
    const [predictionData, setPredictionData] = useState(null);
    const [imagePath, setImagePath] = useState(null);
    const [model, setModel] = useState(null);
    const [loading, setLoading] = useState(false);
    const [xaiImagePath, setXaiImagePath] = useState(null);
    const [emotion, setEmotion] = useState(null);

    useEffect(() => {
        // Retrieve prediction data from URL parameters
        const urlParams = new URLSearchParams(window.location.search);
        const data = urlParams.get('emptionData');
        const path = urlParams.get('imagePath');
        const modelName = urlParams.get("model");

        // Parse and set prediction data
        if (data) {
            const parsedData = JSON.parse(decodeURIComponent(data));
            setPredictionData(parsedData);
            console.log("Loaded prediction data: ", parsedData);
            setEmotion(parsedData.emotion);
            //setAccuracy(parsedData.probability.toFixed(2)); 
            toast.warn('While our machine learning model strives for accuracy, there is a possibility of error in its predictions. Therefore, if you have any doubts regarding the prediction, we strongly advise consulting with a qualified medical professional. Thank you for your understanding and cooperation.', { autoClose: 6000, className: 'custom-toast-message' });
        }

        if (path) {
            setImagePath(decodeURIComponent(path));
            setModel(decodeURIComponent(modelName))
        }
    }, []);

    const limecheck = async () => {
        setLoading(true);
        try {
            const response = await fetch(`http://localhost:8000/xai-lime?filepath=${imagePath}&selected_model=${model}`);
            if (response.ok) {
                const data = await response.json();
                console.log("lime image:", data.xai_lime_path);
                setXaiImagePath(data.xai_lime_path);
                if (data.xai_lime_path != null) {
                    toast.success('XAI LIME interpretable explanation successful!', { autoClose: 1500 });
                }
            }
        } catch (error) {
            console.error("Error: " + error);
        } finally {
            setLoading(false);
        }
    }

    const gradCamCheck = async () => {
        setLoading(true);
        try {
            const response = await fetch(`http://localhost:8000/emotion-xai-gradcam?filepath=${imagePath}&selected_model=${model}`);
            if (response.ok) {
                const data = await response.json();
                console.log(data);
                setXaiImagePath(data.xai_gradCAM_path);
                if (data.xai_gradCAM_path != null) {
                    toast.success('XAI GradCam interpretable explanation successful!', { autoClose: 1500 });
                }
            }
        } catch (error) {
            console.error("Error: " + error);
        } finally {
            setLoading(false);
        }
    }

    const backButton = () => {
        setXaiImagePath(null);
    }

    return (
        <div>
            <nav className="navbar">
                <div className="container d-flex justify-content-center" style={{ gap: "30px" }}>
                    <Link className="navbar-brand" to="/" style={{ fontSize: "30px", fontWeight: "bold " }}>Aunite</Link>
                    <Link className='nav-link' style={{ fontSize: "20px" }} to="/">Home</Link>
                    <Link className='nav-link' style={{ fontSize: "20px" }} to="/guide">Guide</Link>
                </div>
            </nav>
            <div className="container">
                {loading && <LoadingOverlay />}
                <div className="row">
                    <div className="col-lg-6" >
                        <div className="heading">
                            <div className="title">
                                <p>Emotion :</p>
                            </div>
                            <div className="sub-title">
                                <p>Child Facial Emotion is <span style={{ fontWeight: 800 }}>{emotion}</span><br /></p>
                            </div>

                        </div>
                    </div>
                    <div className="col-lg-4">
                        <div>
                            {xaiImagePath ? (
                                <>
                                    <img className='xai-image' src={xaiImagePath} alt="LIME Image" />
                                    <div className="sub-title">
                                        <p style={{ fontSize: '20px', marginTop: '10px', textAlign: 'center' }}>XAI Interpretable explanation</p>
                                    </div>
                                    <button onClick={backButton} className="btn btn-info btn-back">Back</button>
                                </>
                            ) : (
                                (
                                    <>

                                        <button onClick={limecheck} className="btn btn-info btn-xai-lime">
                                            For XAI LIME Output
                                        </button>
                                        <button onClick={gradCamCheck} className="btn btn-xai-gradcam">
                                            For XAI GradCam Output
                                        </button>


                                    </>
                                )
                            )}
                            <ToastContainer />
                        </div>

                    </div>
                    <div className="col-lg-2">

                    </div>
                </div>
            </div>
        </div>
    );
};

export default Emotion_Prediction;