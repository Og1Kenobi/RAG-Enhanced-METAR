# ✈️ AviateGPT AI - RAG Enhanced METAR

**Local LLM-Powered Aviation Weather Intelligence**

A privacy-focused aviation weather tool that fetches real-time data from FAA/NOAA sources and uses a local Ollama LLM to generate clear, pilot-friendly briefings.

---

## Current Capabilities

- Live METAR fetching from official FAA sources
- Advanced regex-based METAR decoding (wind, visibility, ceiling, temperature/dewpoint, altimeter, flight rules)
- Real FAA airport runway data
- Accurate headwind/crosswind calculations for each runway
- Best landing runway recommendation
- Retrieval of NOTAMs and AIRMETs/SIGMETs
- Local AI Pilot Briefings via Ollama

---

## How It Works

**Hybrid Deterministic + LLM Architecture:**

1. **Data Retrieval**  
   Pulls raw METARs, airport runway information, NOTAMs, and AIRMETs/SIGMETs directly from FAA `aviationweather.gov` APIs.

2. **Deterministic Parsing Engine**  
   Uses robust regex patterns and trigonometry for reliable extraction of weather parameters and precise runway wind component calculations.

3. **Local LLM Augmentation**  
   Sends targeted prompts to a locally running Ollama model to generate natural language pilot briefings and decode complex SIGMETs/AIRMETs.

4. **Streamlit Interface**  
   Clean, expandable dashboard designed for pilots.

**Note:** Internet connection is **required** to fetch weather data. The LLM (Ollama) runs completely locally/offline after the model is downloaded.

---

## Model Support

The app is **not limited** to any specific model. It was primarily tested with `qwen2.5-coder:14b`, but it works with **any model** you have in Ollama.

To change the model, edit this line in `src/decoder/weather_parser.py`:

```python
self.model_name = "qwen2.5-coder:14b"   # ← Change to llama3.1:8b, mistral, phi3, gemma2, etc.

Installation
1. Install Ollama and pull a model
Bashollama run qwen2.5-coder:14b
2. Install the App
Bashgit clone https://github.com/Og1Kenobi/RAG-Enhanced-METAR.git
cd RAG-Enhanced-METAR

pip install -r requirements.txt
3. Run the Application
Bashstreamlit run app.py
Open your browser and go to http://localhost:8501

Project Structure
textRAG-Enhanced-METAR/
├── app.py                    # Streamlit frontend
├── src/
│   └── decoder/
│       └── weather_parser.py # Core logic, parsing, runway analysis, LLM calls
├── requirements.txt
└── README.md

Author: Jonathan Douglas (@Og1Kenobi)
Built for real-world General Aviation use with emphasis on accuracy and privacy.
