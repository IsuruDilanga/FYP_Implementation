import { BrowserRouter as Router, Route, Routes } from 'react-router-dom';
import Home from './components/Home/Home';
import ASD_Prediction from './components/ASD_Prediction/ASD_Prediction'
import React from 'react';

function App() {
  return (
    <div className="App">
      <Router>
        <div className="pages">
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/asdprediction" element={<ASD_Prediction />} />
            {/* Define more routes as needed */}
          </Routes>
        </div>
      </Router>
    </div>
  );
}

export default App;
