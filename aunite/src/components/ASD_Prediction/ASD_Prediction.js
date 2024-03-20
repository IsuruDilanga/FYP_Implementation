import React, { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import './ASD_Prediction.css';
import LoadingOverlay from "../LoadingOverlay/LoadingOverlay";
import { toast, ToastContainer } from 'react-toastify';
import 'react-toastify/dist/ReactToastify.css';

const ASD_Prediction = () => {
    const [predictionData, setPredictionData] = useState(null);
    const [imagePath, setImagePath] = useState(null);
    const [model, setModel] = useState(null);
    const [loading, setLoading] = useState(false);
    const [isASD, setASD] = useState(null);
    const [accuracy, setAccuracy] = useState(null);
    const [xaiImagePath, setXaiImagePath] = useState(null);

    useEffect(() => {
        // Retrieve prediction data from URL parameters
        const urlParams = new URLSearchParams(window.location.search);
        const data = urlParams.get('predictionData');
        const path = urlParams.get('imagePath');
        const modelName = urlParams.get("model");

        // Parse and set prediction data
        if (data) {
            const parsedData = JSON.parse(decodeURIComponent(data));
            setPredictionData(parsedData);
            console.log("Loaded prediction data: ", parsedData);
            setAccuracy(parsedData.prediction.toFixed(2)); // Set accuracy to 2 decimal points
            toast.warn('While our machine learning model strives for accuracy, there is a possibility of error in its predictions. Therefore, if you have any doubts regarding the prediction, we strongly advise consulting with a qualified medical professional. Thank you for your understanding and cooperation.', { autoClose: 6000,className: 'custom-toast-message' });
        }

        if (path) {
            setImagePath(decodeURIComponent(path));
            setModel(decodeURIComponent(modelName))
        }
    }, []);

    useEffect(() => {
        // Check if predictionData has been set and update isASD accordingly
        if (predictionData) {
            setASD(predictionData.isASD ? 'true' : 'false');
        }
    }, [predictionData]);

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
            const response = await fetch(`http://localhost:8000/xai-gradcam?filepath=${imagePath}&selected_model=${model}`);
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

    const checkEmotion = async () => {
        setLoading(true);
        try {
            const response = await fetch(`http://localhost:8000/predictEmotoin?filepath=${imagePath}&selected_model=${model}`);
            if (response.ok) {
                const data = await response.json();
                console.log('Prediction successful:', data);
                setPredictionData(data);
                console.log("data", data);
                console.log("image Path", imagePath);
                console.log("model", model);
                window.location.href = `/emotionprediction?emptionData=${encodeURIComponent(JSON.stringify(data))}&imagePath=${encodeURIComponent(imagePath)}&model=${encodeURIComponent(model)}`; 
            } else {
                console.error('Failed to predict ASD:', response.statusText);
                // Handle error
            }
        }catch(error){

        } finally{
            setLoading(false);
        }
        // console.log("checking emotion...");
        // console.log("imagePath : ", imagePath);
        // console.log("Model :", model);
    }

    return (
        <div>
            <nav className="navbar">
                <div className="container">
                    <a className="navbar-brand">Aunite</a>
                    <form className="d-flex" role="search">
                        <input className="form-control me-2" type="search" placeholder="Search" aria-label="Search" />
                        <button className="btn btn-outline-success" type="submit">Search</button>
                    </form>
                </div>
            </nav>
            <div className="container">
                {loading && <LoadingOverlay />}
                <div className="row">
                    <div className="col-lg-8" >
                        <div className="heading">
                            <div className="title">
                                <p>Result:</p>
                            </div>
                            <div className="sub-title">
                                <p>Child Image Prediction as <span style={{ fontWeight: 800, color: isASD === 'true' ? 'red' : 'black' }}>{isASD === 'true' ? 'Autism Spectrum Disorder' : 'not Autism Spectrum Disorder'}</span><br />
                                    Accuracy of: <span style={{ fontWeight: 800 }}>{accuracy}</span></p>
                            </div>
                            {isASD === 'true' && (
                                <>
                                    <button onClick={checkEmotion} type="button" className="btn btn-warning btn-emotion" style={{fontSize:'20px'}}>Check Clid Emotion</button>
                                </>
                            )}
                            
                        </div>
                    </div>
                    <div className="col-lg-4">
                        <div>
                            {xaiImagePath ? (
                                <>
                                    <img className='xai-image' src={xaiImagePath} alt="LIME Image" />
                                    <div className="sub-title">
                                        <p style={{ fontSize: '20px', marginTop: '10px', textAlign:'center' }}>XAI Interpretable explanation</p>
                                    </div>
                                    <button onClick={backButton} className="btn btn-info btn-back">Back</button>
                                </>
                            ) : (
                                isASD === 'true' && (
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
                            <ToastContainer/>
                        </div>
                        
                    </div>
                </div>
            </div>
        </div>
    );
};

export default ASD_Prediction;



////////////////////
// import React, { useEffect, useState } from 'react';
// import { Link } from 'react-router-dom';
// import './ASD_Prediction.css'
// import LoadingOverlay from "../LoadingOverlay/LoadingOverlay";
// import { toast, ToastContainer } from 'react-toastify';
// import 'react-toastify/dist/ReactToastify.css';

// const ASD_Prediction = () => {
//     const [predictionData, setPredictionData] = useState(null);
//     const [imagePath, setImagePath] = useState(null);
//     const [model, setModel] = useState(null);
//     const [loading, setLoading] = useState(false);
//     const [isASD, setASD] = useState(null);
//     const [accuracy, setAccuracy] = useState(null);
//     const [xaiImagePath, setxaiImagePath] = useState(null);

//     useEffect(() => {
//         // Retrieve prediction data from URL parameters
//         const urlParams = new URLSearchParams(window.location.search);
//         const data = urlParams.get('predictionData');
//         const path = urlParams.get('imagePath');
//         const modelName = urlParams.get("model");

//         // Parse and set prediction data
//         if (data) {
//             const parsedData = JSON.parse(decodeURIComponent(data));
//             setPredictionData(parsedData);
//             console.log("Loaded prediction data: ", parsedData);
//             setAccuracy(parsedData.prediction.toFixed(2)); // Set accuracy to 2 decimal points
//             toast.warn('While our machine learning model strives for accuracy, there is a possibility of error in its predictions. Therefore, if you have any doubts regarding the prediction, we strongly advise consulting with a qualified medical professional. Thank you for your understanding and cooperation.', { autoClose: 6000,className: 'custom-toast-message' });

//         }

//         if (path) {
//             setImagePath(decodeURIComponent(path));
//             setModel(decodeURIComponent(modelName))
//         }
//     }, []);

//     useEffect(() => {
//         // Check if predictionData has been set and update isASD accordingly
//         if (predictionData) {
//             setASD(predictionData.isASD ? 'true' : 'false');
//         }
//     }, [predictionData]);

//     const limecheck = async () => {
//         setLoading(true);
//         try {
//             const response = await fetch(`http://localhost:8000/xai-lime?filepath=${imagePath}&selected_model=${model}`);
//             if (response.ok) {
//                 const data = await response.json();
//                 console.log("lime image:", await data.xai_lime_path);
//                 setxaiImagePath(await data.xai_lime_path);
//                 if(await data.xai_lime_path != null){
//                     toast.success('XAI LIME interpretable explanation successful!', { autoClose: 1500 });
//                 }
//             }

//         } catch (error) {
//             console.error("Error: " + error);
//         } finally {
//             setLoading(false);
//         }
//     }

//     const gradCamCheck = async () => {
//         try {
//             const response = await fetch(`http://localhost:8000/xai-gradcam?filepath=${imagePath}&selected_model=${model}`);
//             if (response.ok) {
//                 const data = await response.json();
//                 console.log(await data);
//                 setxaiImagePath(await data.xai_gradCAM_path);
//                 if(await data.xai_gradCAM_path != null){
//                     toast.success('XAI GradCam interpretable explanation successful!', { autoClose: 1500 });
//                 }            }
//         } catch (error) {
//             console.error("Error: " + error);
//         }
//     }

//     const backButton = () => {
//         setxaiImagePath(null);
//     }

//     return (
//         <div>
//             <nav className="navbar">
//                 <div className="container">
//                     <a className="navbar-brand">Aunite</a>
//                     <form className="d-flex" role="search">
//                         <input className="form-control me-2" type="search" placeholder="Search" aria-label="Search" />
//                         <button className="btn btn-outline-success" type="submit">Search</button>
//                     </form>
//                 </div>
//             </nav>
//             <div className="container">
//                 {loading && <LoadingOverlay />}
//                 <div className="row">
//                     <div className="col-lg-8" >
//                         <div className="heading">
//                             <div className="title">
//                                 <p>Result:</p>
//                             </div>
//                             <div className="sub-title">
//                                 <p>Child Image Prediction as <span style={{ fontWeight: 800, color: isASD === 'true' ? 'red' : 'black' }}>{isASD === 'true' ? 'Autism Spectrum Disorder' : 'not Autism Spectrum Disorder'}</span><br />
//                                     Accuracy of: <span style={{ fontWeight: 800 }}>{accuracy}</span></p>
//                                     {xaiImagePath.includes && xaiImagePath.includes('lime') && (
//                                         <p>XAI LIME graphical interpretable explanation</p>
//                                     )}
//                                     {xaiImagePath.includes && xaiImagePath.includes('gradcam') && (
//                                         <p>XAI GradCAM graphical interpretable explanation</p>
//                                     )}
//                                     {/* <p style={{ fontSize: '20px', marginTop: '10px' }}>Interpretable explanation</p> */}
//                             </div>
//                         </div>
//                     </div>
//                     <div className="col-lg-4">
//                         <div>
//                             {xaiImagePath ? (
//                                     <>
//                                         <img className='xai-image' src={xaiImagePath} alt="LIME Image" />
//                                         <div className="sub-title">
                                            
//                                         </div>
//                                         <button onClick={backButton} className="btn btn-info btn-back">Back</button>
//                                     </>
//                                 ) : (
//                                     isASD === 'true' && (
//                                         <>
//                                             <button onClick={limecheck} className="btn btn-info btn-xai-lime">
//                                                 For XAI LIME Output
//                                             </button>
//                                             <button onClick={gradCamCheck} className="btn btn-xai-gradcam">
//                                                 For XAI GradCam Output
//                                             </button>
//                                         </>
//                                     )
//                                 )}
                            
//                             <ToastContainer/>
//                         </div>
//                     </div>
//                 </div>
//             </div>
//         </div>
//     );
// };

// export default ASD_Prediction;




//////////////////
// import React, { useEffect, useState } from 'react';
// import { Link } from 'react-router-dom';
// import './ASD_Prediction.css'
// import LoadingOverlay from "../LoadingOverlay/LoadingOverlay";
// import { toast, ToastContainer } from 'react-toastify';
// import 'react-toastify/dist/ReactToastify.css';

// const ASD_Prediction = () => {
//     const [predictionData, setPredictionData] = useState(null);
//     const [imagePath, setImagePath] = useState(null);
//     const [model, setModel] = useState(null);
//     const [loading, setLoading] = useState(false);
//     const [isASD, setASD] = useState(null);
//     const [accuracy, setAccuracy] = useState(null);
//     const [limeImagePath, setLimeImagePath] = useState(null);

//     useEffect(() => {
//         // Retrieve prediction data from URL parameters
//         const urlParams = new URLSearchParams(window.location.search);
//         const data = urlParams.get('predictionData');
//         const path = urlParams.get('imagePath');
//         const modelName = urlParams.get("model");

//         // Parse and set prediction data
//         if (data) {
//             const parsedData = JSON.parse(decodeURIComponent(data));
//             setPredictionData(parsedData);
//             console.log("Loaded prediction data: ", parsedData);
//             setAccuracy(parsedData.prediction.toFixed(2)); // Set accuracy to 2 decimal points
//         }

//         if (path) {
//             setImagePath(decodeURIComponent(path));
//             setModel(decodeURIComponent(modelName))
//         }
//     }, []);

//     useEffect(() => {
//         // Check if predictionData has been set and update isASD accordingly
//         if (predictionData) {
//             setASD(predictionData.isASD ? 'true' : 'false');
//         }
//     }, [predictionData]);

//     const limecheck = async () => {
//         setLoading(true);
//         try {
//             const response = await fetch(`http://localhost:8000/xai-lime?filepath=${imagePath}&selected_model=${model}`);
//             if (response.ok) {
//                 const data = await response.json();
//                 console.log("lime image:", await data.xai_lime_path);
//                 setLimeImagePath(await data.xai_lime_path);
//                 toast.success('XAI LIME interpretable explanation successful!', { autoClose: 1500 });
//             }

//         } catch (error) {
//             console.error("Error: " + error);
//         } finally {
//             setLoading(false);
//         }
//     }

//     const gradCamCheck = async () => {
//         try {
//             const response = await fetch(`http://localhost:8000/xai-lime?filepath=${imagePath}&model=${model}`);
//             if (response.ok) {
//                 const data = await response.json();
//                 console.log(await data);
//             }
//         } catch (error) {
//             console.error("Error: " + error);
//         }
//     }

//     return (
//         <div>
//             <nav className="navbar">
//                 <div className="container">
//                     <a className="navbar-brand">Aunite</a>
//                     <form className="d-flex" role="search">
//                         <input className="form-control me-2" type="search" placeholder="Search" aria-label="Search" />
//                         <button className="btn btn-outline-success" type="submit">Search</button>
//                     </form>
//                 </div>
//             </nav>
//             <div className="container">
//                 {loading && <LoadingOverlay />}
//                 <div className="row">
//                     <div className="col-lg-8" >
//                         <div className="heading">
//                             <div className="title">
//                                 <p>Result:</p>
//                             </div>
//                             <div className="sub-title">
//                                 <p>Child Image Prediction as <span style={{ fontWeight: 800, color: isASD === 'true' ? 'red' : 'black' }}>{isASD === 'true' ? 'Autism Spectrum Disorder' : 'not Autism Spectrum Disorder'}</span><br />
//                                     Accuracy of: <span style={{ fontWeight: 800 }}>{accuracy}</span></p>
//                             </div>
//                         </div>
//                     </div>
//                     <div className="col-lg-4">
//                         <div>
//                             {isASD === 'true' && (
//                                 <>
//                                     <button onClick={limecheck} className="btn btn-info btn-xai-lime">
//                                         For XIA LIME Output
//                                     </button>
//                                     <button onClick={gradCamCheck} className="btn btn-xai-gradcam">
//                                         For XIA GradCam Output
//                                     </button>
//                                 </>
//                             )}
//                             <ToastContainer />
//                         </div>
//                     </div>
//                 </div>
//             </div>
//         </div>
//     );
// };

// export default ASD_Prediction;


////////////////////////
// import React, { useEffect, useState } from 'react';
// import { Link } from 'react-router-dom';
// import './ASD_Prediction.css'
// import LoadingOverlay from "../LoadingOverlay/LoadingOverlay";
// import { toast, ToastContainer } from 'react-toastify';
// import 'react-toastify/dist/ReactToastify.css';

// const ASD_Prediction = () => {
//     const [predictionData, setPredictionData] = useState(null);
//     const [imagePath, setImagePath] = useState(null);
//     const [model, setModel] = useState(null);
//     const [loading, setLoading] = useState(false);
//     const [isASD, setASD] = useState(null);
//     const [accuracy, setAccuracy] = useState(null);

//     useEffect(() => {
//         // Retrieve prediction data from URL parameters
//         const urlParams = new URLSearchParams(window.location.search);
//         const data = urlParams.get('predictionData');
//         const path = urlParams.get('imagePath');
//         const modelName = urlParams.get("model");

//         // Parse and set prediction data
//         if (data) {
//             const parsedData = JSON.parse(decodeURIComponent(data));
//             setPredictionData(parsedData);
//             console.log("Loaded prediction data: ", parsedData);
//             setAccuracy(parsedData.prediction.toFixed(2)); // Set accuracy to 2 decimal points
//         }

//         if (path) {
//             setImagePath(decodeURIComponent(path));
//             setModel(decodeURIComponent(modelName))
//         }
//     }, []);

//     useEffect(() => {
//         // Check if predictionData has been set and update isASD accordingly
//         if (predictionData) {
//             setASD(predictionData.isASD ? 'true' : 'false');
//         }
//     }, [predictionData]);

//     const limecheck = async () => {
//         setLoading(true);
//         try {
//             const response = await fetch(`http://localhost:8000/xai-lime?filepath=${imagePath}&selected_model=${model}`);
//             if (response.ok) {
//                 const data = await response.json();
//                 console.log(await data);
//                 toast.success('XAI LIME interpritable explanation sucessfull!', { autoClose: 1500 });
//             }

//         } catch (error) {
//             console.error("Error: " + error);
//         } finally {
//             setLoading(false);
//         }
//     }

//     const gradCamCheck = async () => {
//         try {
//             const response = await fetch(`http://localhost:8000/xai-lime?filepath=${imagePath}&model=${model}`);
//             if (response.ok) {
//                 const data = await response.json();
//                 console.log(await data);
//             }
//         } catch (error) {
//             console.error("Error: " + error);
//         }
//     }

//     return (
//         <div>
//             <nav className="navbar">
//                 <div className="container">
//                     <a className="navbar-brand">Aunite</a>
//                     <form className="d-flex" role="search">
//                         <input className="form-control me-2" type="search" placeholder="Search" aria-label="Search" />
//                         <button className="btn btn-outline-success" type="submit">Search</button>
//                     </form>
//                 </div>
//             </nav>
//             <div className="container">
//                 {loading && <LoadingOverlay />}
//                 <div className="row">
//                     <div className="col-lg-8" >
//                         <div className="heading">
//                             <div className="title">
//                                 <p>Result:</p>
//                             </div>
//                             <div className="sub-title">
//                                 <p>Child Image Prediction as <span style={{ fontWeight: 800, color: isASD === 'true' ? 'red' : 'black' }}>{isASD === 'true' ? 'Autism Spectrum Disorder' : 'not Autism Spectrum Disorder'}</span><br />
//                                     Accuracy of: <span style={{ fontWeight: 800 }}>{accuracy}</span></p>
//                             </div>
//                         </div>
//                     </div>
//                     <div className="col-lg-4">
//                         <div>
                            
//                             <button onClick={limecheck} className="btn btn-info btn-xai-lime">
//                                 For XIA LIME Output
//                             </button>
//                             <button onClick={gradCamCheck} className="btn btn-xai-gradcam">
//                                 For XIA GradCam Output
//                             </button>
//                             <ToastContainer />
//                         </div>
//                     </div>
//                 </div>
//             </div>
//         </div>
//     );
// };

// export default ASD_Prediction;

////////////////
// import React, { useEffect, useState } from 'react';
// import { Link } from 'react-router-dom';
// import './ASD_Prediction.css'
// import LoadingOverlay from "../LoadingOverlay/LoadingOverlay";
// import { toast, ToastContainer } from 'react-toastify';
// import 'react-toastify/dist/ReactToastify.css';

// const ASD_Prediction = () => {
//     const [predictionData, setPredictionData] = useState(null);
//     const [imagePath, setImagePath] = useState(null);
//     const [model, setModel] = useState(null);
//     const [loading, setLoading] = useState(false);
//     const [isASD, setASD] = useState(null);
//     const [accuracy, setAccuracy] = useState(null);

//     let accuracyDecimal;

//     useEffect(() => {
//         // Retrieve prediction data from URL parameters
//         const urlParams = new URLSearchParams(window.location.search);
//         const data = urlParams.get('predictionData');
//         const path = urlParams.get('imagePath');
//         const modelName = urlParams.get("model");

//         // Parse and set prediction data
//         if (data) {
//             const parsedData = JSON.parse(decodeURIComponent(data));
//             setPredictionData(parsedData);
//             console.log("Loaded prediction data: ", parsedData);
//             setAccuracy(parsedData.prediction);
//         }

//         if (path) {
//             setImagePath(decodeURIComponent(path));
//             setModel(decodeURIComponent(modelName))
//         }
//     }, []);

//     if (accuracy = !null) {
//         accuracyDecimal = accuracy.toFixed(2);
//     }

//     useEffect(() => {
//         // Check if predictionData has been set and update isASD accordingly
//         if (predictionData) {
//             setASD(predictionData.isASD ? 'true' : 'false');
//         }
//     }, [predictionData]);

//     const limecheck = async () => {
//         setLoading(true);
//         try {
//             const response = await fetch(`http://localhost:8000/xai-lime?filepath=${imagePath}&model=${model}`);
//             if (response.ok) {
//                 const data = await response.json();
//                 console.log(await data);
//                 toast.success('XAI LIME interpritable explanation sucessfull!', { autoClose: 1500 });
//             }

//         } catch (error) {
//             console.error("Error: " + error);
//         } finally {
//             setLoading(false);
//         }
//     }

//     const gradCamCheck = async () => {

//         try {
//             const response = await fetch(`http://localhost:8000/xai-lime?filepath=${imagePath}&model=${model}`);
//             if (response.ok) {
//                 const data = await response.json();
//                 console.log(await data);

//             }

//         } catch (error) {
//             console.error("Error: " + error);
//         }
//     }

//     return (
//         <div>
//             <nav className="navbar">
//                 <div className="container">
//                     <a className="navbar-brand">Aunite</a>
//                     <form className="d-flex" role="search">
//                         <input className="form-control me-2" type="search" placeholder="Search" aria-label="Search" />
//                         <button className="btn btn-outline-success" type="submit">Search</button>
//                     </form>
//                 </div>
//             </nav>
//             <div className="container">
//                 {loading && <LoadingOverlay />}
//                 <div className="row">
//                     <div className="col-lg-8" >
//                         <div className="heading">
//                             <div className="title">
//                                 <p>Result:</p>
//                             </div>
//                             <div className="sub-title">
//                                 <p>Child Image Prediction as <span style={{ fontWeight: 800, color: isASD ? 'red' : 'black' }}>{isASD ? 'Autism Spectrum Disorder' : 'not Autism Spectrum Disorder'}</span><br></br>
//                                     Accuracy of: <span style={{ fontWeight: 800 }}>{accuracyDecimal}</span></p>
//                             </div>
//                         </div>
//                     </div>
//                     <div className="col-lg-4">
//                         <div>
//                             <button onClick={limecheck} className="btn btn-info btn-xai-lime">
//                                 For XIA LIME Output
//                             </button>
//                             <button onClick={gradCamCheck} className="btn btn-xai-gradcam">
//                                 For XIA GradCam Output
//                             </button>
//                             <ToastContainer />
//                         </div>
//                     </div>
//                 </div>
//             </div>
//         </div>
//     );
// };

// export default ASD_Prediction;
