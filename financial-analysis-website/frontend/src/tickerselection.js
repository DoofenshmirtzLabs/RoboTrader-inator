import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import './App.css';

const TickerSelection = () => {
  const [ticker, setTicker] = useState('');
  const [responseMessage, setResponseMessage] = useState('');
  const [graphUrls, setGraphUrls] = useState([]);
  const [loading, setLoading] = useState(false);
  const graphSectionRef = useRef(null);

  const allowedTickers = ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS",
    "SBIN.NS", "HINDUNILVR.NS", "ITC.NS", "LT.NS", "KOTAKBANK.NS",
    "AXISBANK.NS", "BAJFINANCE.NS", "WIPRO.NS", "BHARTIARTL.NS", "ASIANPAINT.NS",
    "HCLTECH.NS", "MARUTI.NS", "SUNPHARMA.NS", "ULTRACEMCO.NS", "TITAN.NS",
    "POWERGRID.NS", "NTPC.NS", "INDUSINDBK.NS", "ONGC.NS", "ADANIENT.NS",
    "JSWSTEEL.NS", "TATAMOTORS.NS", "ADANIGREEN.NS", "ADANIPORTS.NS", "COALINDIA.NS",
    "DIVISLAB.NS", "GRASIM.NS", "BPCL.NS", "SHREECEM.NS", "TECHM.NS",
    "BRITANNIA.NS", "CIPLA.NS", "DRREDDY.NS", "EICHERMOT.NS", "HEROMOTOCO.NS",
    "HDFCLIFE.NS", "ICICIPRULI.NS", "IOC.NS", "M&M.NS", "NESTLEIND.NS",
    "PIDILITIND.NS", "TATASTEEL.NS", "UPL.NS", "SBILIFE.NS", "BAJAJFINSV.NS"]

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!ticker || !allowedTickers.includes(ticker)) {
      setResponseMessage('Invalid ticker symbol. Please enter a valid one.');
      return;
    }

    setLoading(true);
    setResponseMessage('');
    setGraphUrls([]);

    try {
      const response = await axios.post('http://127.0.0.1:5000/analyze', { ticker });
      if (response.data.graphUrls) {
        setGraphUrls(response.data.graphUrls);
        setResponseMessage(response.data.message);
      } else {
        setResponseMessage(response.data.error || 'No graphs generated.');
      }
    } catch (error) {
      setResponseMessage(error.response?.data?.error || 'Error occurred. Please check the server logs.');
    } finally {
      setLoading(false);
    }
  };
  useEffect(() => {
    if (graphUrls.length > 0 && graphSectionRef.current) {
      setTimeout(() => {
        const topOffset =
          graphSectionRef.current.getBoundingClientRect().top +
          window.pageYOffset -
          100; // adjust offset if you have header

        window.scrollTo({
          top: topOffset,
          behavior: 'smooth',
        });
      }, 300); // slight delay to ensure DOM updates
    }
  }, [graphUrls]);
  

  return (
    <div className="container">
      {/* Welcome Message */}
      <div className="welcome-message">
        <h1>Welcome to Robo-Trader!</h1>
        <p>Welcome to the Stock Analysis Platform! Here, you can explore the top-performing trading strategies recommended by our advanced genetic algorithm. Simply enter the ticker symbol of your favorite or most interesting stocks, and our algorithm will analyze historical data to generate insightful performance graphs. Visualize how the recommended strategies have performed over time, compare trends, and gain actionable insights to make informed investment decisions. Let’s uncover the best strategies tailored to your stock interests!</p>
      </div>
      <h1>Select a Ticker for Analysis</h1>
      <form onSubmit={handleSubmit} className="form-container">
        <input
          type="text"
          value={ticker}
          onChange={(e) => setTicker(e.target.value.toUpperCase())}
          placeholder="Enter Ticker Symbol (e.g., INFY.NS)"
          className="input-field"
        />
        <button type="submit" className="submit-button">Analyze</button>
      </form>

      {loading && <p className="loading-text">Loading... Please wait while we analyze the data.</p>}
      {responseMessage && <p className="response-message">{responseMessage}</p>}

      {graphUrls.length > 0 && (
        <div className="graph-section" ref={graphSectionRef}>
          {/* First graph displayed solo in center */}
          <div className="graph-solo">
            <img src={graphUrls[0]} alt="Graph 1" />
          </div>

          {/* Remaining graphs in pairs with strategy numbers */}
          <div className="graph-grid">
            {graphUrls.slice(1).map((url, index) => (
              <React.Fragment key={index}>
                {index % 2 === 0 && <h2 className="strategy-title">Strategy {index / 2 + 1}</h2>}
                <div className="graph-card">
                  <img src={url} alt={`Graph ${index + 2}`} />
                </div>
              </React.Fragment>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

export default TickerSelection;







