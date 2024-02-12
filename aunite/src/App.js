import { BrowserRouter as Router, Route, Routes } from 'react-router-dom';
import Home from './components/Home/Home';
import React from 'react';

function App() {
  return (
    <div className="App">
      <Router>
        <div className="pages">
          <Routes>
            <Route path="/" element={<Home />} />
            {/* Define more routes as needed */}
          </Routes>
        </div>
      </Router>
    </div>
  );
}

export default App;
