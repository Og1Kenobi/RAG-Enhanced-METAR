import sys
from pathlib import Path
from datetime import datetime

current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

import streamlit as st
from src.decoder.weather_parser import WeatherParser

st.set_page_config(page_title="AviateGPT AI", page_icon="✈️", layout="wide")

st.markdown("""
<style>
    .stApp { background: linear-gradient(135deg, #f8fafc 0%, #e0f2fe 100%); }
    .main .block-container { padding-top: 2rem; }
    h1, h2, h3 { color: #1e40af; }
    .stButton>button { background: linear-gradient(90deg, #2563eb, #3b82f6); color: white; border-radius: 12px; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

st.title("✈️ AviateGPT AI")
st.caption("**RAG-Enhanced METAR Decoder** • Live FAA Data")

parser = WeatherParser()

with st.sidebar:
    st.header("Flight Parameters")
    airport_code = st.text_input("ICAO Airport Code", value="KXNA")
    submit_button = st.button("Pull NOAA Data & Analyze", type="primary", use_container_width=True)

    st.divider()
    if st.button("🧠 Aviation Regulations Brain (FAR/AIM)", use_container_width=True):
        st.switch_page("pages/1_Aviation_Brain.py")
    st.caption("Local LLM • Live FAA Sources")

if submit_button and airport_code:
    with st.spinner("Fetching live METAR + NOTAMs..."):
        raw_feed, is_current = parser.fetch_live_metar(airport_code)

    if "ERROR" in raw_feed:
        st.error(raw_feed)
    else:
        data = parser.decode_metar(raw_feed)
        data["is_current"] = is_current

        if data.get("status") == "Error":
            st.error(data.get("message"))
        else:
            if not is_current:
                st.warning("⚠️ Using most recent available METAR (may be outdated)")

            tab1, tab2, tab3, tab4 = st.tabs(["📡 Current Conditions", "🛫 Runway Analysis", "🌩️ AIRMETs/SIGMETs", "⚠️ NOTAMs"])

            with tab1:
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.subheader("Raw METAR")
                    st.code(data.get("raw_feed"), language="text")

                with col2:
                    with st.spinner("Generating briefing..."):
                        briefing = parser.fetch_ai_briefing(data.get("raw_feed"))
                    st.subheader("Pilot Briefing")
                    st.markdown(briefing)

                rules = data.get("flight_rules", "VFR")
                if rules == "IFR":
                    st.error(f"❌ IFR Conditions ({rules})")
                elif rules == "MVFR":
                    st.warning(f"⚠️ Marginal VFR ({rules})")
                else:
                    st.success(f"✅ VFR Active ({rules})")

                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    st.metric("Station", data.get("airport", airport_code).upper())
                    wind_str = f"{data.get('wind_dir')}° @ {data.get('wind_speed_kt', 0)} KT"
                    if data.get('wind_gust_kt'): wind_str += f" G{data.get('wind_gust_kt')}KT"
                    st.metric("Surface Winds", wind_str)
                with c2:
                    st.metric("Visibility", data.get("visibility"))
                    st.metric("Ceiling", f"{data.get('ceiling_ft', '---'):,} ft")
                with c3:
                    st.metric("Temperature", f"{data.get('temp_c')}°C")
                    st.metric("Dewpoint", f"{data.get('dew_c')}°C")
                with c4:
                    spread = int(data.get("temp_c", 0)) - int(data.get("dew_c", 0))
                    st.metric("Temp Spread", f"{spread}°C")

            with tab2:
                st.subheader("🛫 Runway Wind Analysis")
                calculations = data.get("runway_report", [])
                if calculations:
                    table_data = []
                    for item in calculations:
                        table_data.append({
                            "Runway": item.get('direction'),
                            "Headwind": f"{item.get('headwind', 0)} KT",
                            "Crosswind": f"{item.get('crosswind', 0)} KT",
                            "Best": "✅ Recommended" if item.get("is_best") else ""
                        })
                    st.dataframe(table_data, use_container_width=True, hide_index=True)
                    
                    best = next((item for item in calculations if item.get("is_best")), None)
                    if best:
                        st.success(f"**Recommended: Runway {best.get('direction')}** — {best.get('headwind', 0)} KT headwind")
                else:
                    st.warning("No runway data available.")

            with tab3:
                st.subheader("Active AIRMETs / SIGMETs")
                airmets = data.get("airmets_sigmets", [])
                if airmets:
                    for item in airmets:
                        with st.expander(f"{item.get('airSigmetType')} — {item.get('hazard')}"):
                            st.code(item.get("rawAirSigmet"))
                            st.info(parser.decode_airsigmet_plain_english([item]))
                else:
                    st.success("No active advisories")

            with tab4:
                st.subheader("⚠️ Active NOTAMs")
                notams = data.get("notams", [])
                if notams:
                    for notam in notams:
                        with st.expander(notam.get("notam_id", "NOTAM")):
                            st.code(notam.get("raw_text"))
                else:
                    st.info("No active NOTAMs for this airport at this time.")

            with st.expander("Full JSON Data"):
                st.json(data)

else:
    st.info("Enter ICAO code (KXNA) and click the button.")

st.caption(f"Last updated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")