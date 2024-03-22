import React from 'react';
import { Link } from 'react-router-dom';
import './Guide.css';

const Guide = () => {

    return (
        <div>
            <nav className="navbar">
                <div className="container d-flex justify-content-center" style={{ gap: "30px" }}>
                    <Link className="navbar-brand" to="/" style={{ fontSize: "30px", fontWeight: "bold " }}>Aunite</Link>
                    <Link className='nav-link' style={{ fontSize: "20px" }} to="/">Home</Link>
                    <Link className='nav-link' style={{ fontSize: "20px" }} to="/guide">Guide</Link>
                </div>
            </nav>
            <div className="about-container" style={{ marginTop: '30px', textAlign: 'center' }}>
                <h2>Welcome to our ASD Prediction Website</h2>
                <p>
                    Our website is dedicated to predicting Autism Spectrum Disorder (ASD) in children aged between 3 to 12 years, utilizing advanced Deep Learning technology.
                    We provide insights into ASD prediction, emotional analysis, and interpretable explanation methods, ensuring a better understanding of your child's well-being.
                </p>
                <h3>Facial Image Analysis for ASD Prediction</h3>
                <p>
                    You can update or upload a facial image of the child, and then select the Convolutional Neural Network (CNN) model for ASD detection. The prediction provided can offer insights into whether the child may have ASD or not.
                </p>
                <h3>Understanding Lime and GradCam</h3>
                <p>
                    If ASD is predicted, our website provides interpretable explanation methods, namely 'Lime' and 'GradCam', to help you understand the prediction process.
                </p>
                <ul>
                    <li><strong>Lime Explanation (Lime):</strong> Lime provides a detailed graphical explanation, albeit it may take some time to generate. It illustrates heat points indicating regions where the prediction leans towards ASD, shown in green.</li>
                    <li><strong>Gradient-weighted Class Activation Mapping (GradCam):</strong> GradCam offers a quick image explanation, highlighting regions in red that contribute significantly to the ASD prediction.</li>
                </ul>
                <h3>Detecting Emotions</h3>
                <p>
                    Additionally, if ASD is detected, you have the option to explore the child's emotions. You can choose between 'Lime' and 'GradCam' eXplainable Artificial Intelligence (XAI) libraries for this purpose.
                </p>
                <p>
                    Our website aims to provide you with comprehensive insights into ASD prediction and emotional analysis, ensuring a better understanding of your child's well-being. Remember, while our tools are powerful aids, consulting with a healthcare professional remains essential for accurate diagnosis and support.
                </p>
            </div>

        </div>
    );

}

export default Guide;