import React from "react";
import "./LoadingOverlay.css";

const LoadingOverlay = () => {
    return (
        <div className="loading-overlay">
            <div className="loading-animation"></div>
            <div className="predicting-text">Predicting...</div>
        </div>
    );
};

export default LoadingOverlay;
