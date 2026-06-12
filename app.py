import sys
from pathlib import Path

current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

import streamlit as st
from src.decoder.weather_parser import WeatherParser

st.set_page_config(page_title="AviateGPT AI", layout="wide")
st.title("✈️ AviateGPT AI - RAG Enhanced METAR")
st.caption("Human Readable METAR Decoder & Runway Wind Analyzer • Powered by Qwen2.5-Coder:14B")

parser = WeatherParser()

st.sidebar.markdown("### 🎛️ Flight Parameters")
airport_code = st.sidebar.text_input("Airport Identifier (ICAO):", value="")
submit_button = st.sidebar.button("🛡️ Pull NOAA Data & Auto-Map Airfield", use_container_width=True)

if submit_button:
    with st.spinner("Connecting to live NOAA database..."):
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
                    st.warning("⚠️ **Showing most recent available METAR** (may be expired / outdated)")

                st.markdown("### 📡 Current Raw NOAA Stream")
                st.code(data.get("raw_feed"), language="text")
                
                with st.spinner("Generating AI Pilot Briefing..."):
                    briefing = parser.fetch_ai_briefing(data.get("raw_feed"))
                st.info(f"**Pilot Briefing:** {briefing}")

                rules = data.get("flight_rules", "VFR")
                if rules == "IFR":
                    st.error(f"❌ METEOROLOGICAL ALERT: AIRSPACE RESTRICTED TO INSTRUMENT FLIGHT ({rules})")
                elif rules == "MVFR":
                    st.warning(f"⚠️ MARGINAL VISUAL FLIGHT CONDITIONS IN EFFECT ({rules})")
                else:
                    st.success(f"✅ AIRSPACE OPEN: VISUAL FLIGHT CONDITIONS ACTIVE ({rules})")

                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric(label="Station", value=data.get("airport", airport_code).upper())
                    wind_str = f"{data.get('wind_dir')}° @ {data.get('wind_speed')} KT"
                    if data.get('wind_gust'):
                        wind_str += f" G{data.get('wind_gust')}KT"
                    st.metric(label="Surface Winds", value=wind_str)
                with col2:
                    st.metric(label="Visibility", value=data.get("visibility"))
                    st.metric(label="Cloud Ceiling", value=f"{data.get('ceiling_ft', 99999):,} FT ({data.get('clouds')})")
                with col3:
                    st.metric(label="Temp / Dewpoint", value=f"{data.get('temp_c')}°C / {data.get('dew_c')}°C")
                    st.metric(label="Altimeter Settings", value=f"{data.get('altimeter')} inHg")
                with col4:
                    spread = int(data.get("temp_c", 0)) - int(data.get("dew_c", 0))
                    st.metric(label="Spread", value=f"{spread}°C", delta="Fog Threat" if spread <= 2 else "Dry Air")

                # Runway Report - Clean individual landing directions
                st.markdown("### 🛡️ Automated Runway Wind Component Report")
                calculations = data.get("runway_report", [])
                
                if calculations:
                    markdown_table = "| Landing Direction | Headwind (KT) | Crosswind (KT) | Recommended |\n"
                    markdown_table += "|--------------------|---------------|----------------|-------------|\n"
                    for item in calculations:
                        rec = "**Yes**" if item.get("is_best") else ""
                        dir_display = f"**{item.get('direction', '')}**" if item.get("is_best") else item.get('direction', '')
                        markdown_table += f"| {dir_display} | {item.get('headwind', 0)} | {item.get('crosswind', 0)} | {rec} |\n"
                    st.markdown(markdown_table)

                    # Clear recommendation
                    best_items = [item for item in calculations if item.get("is_best")]
                    if best_items:
                        best = best_items[0]
                        st.success(f"💡 **Recommended: Land on Runway {best.get('direction')}** with **{best.get('headwind', 0)} KT headwind**")
                else:
                    st.warning("No runway data available for this airport.")

                # AIRMETs & SIGMETs (Human Language)
                st.markdown("### 🌩️ Active AIRMETs / SIGMETs (Decoded)")
                airmets = data.get("airmets_sigmets", [])
                if airmets:
                    for i, item in enumerate(airmets[:8]):
                        raw_text = item.get("rawAirSigmet", str(item))
                        title = f"{item.get('airSigmetType', 'ADVISORY')} — {item.get('hazard', 'Hazard')}"
                        with st.expander(f"🌩️ {title}"):
                            st.write("**Raw:**")
                            st.code(raw_text, language="text")
                            st.write("**Plain English Summary:**")
                            summary = parser.decode_airsigmet(raw_text)
                            st.info(summary)
                else:
                    st.success("✅ No active AIRMETs or SIGMETs affecting this area.")

                # NOTAMs
                st.markdown("### ⚠️ Active NOTAMs")
                notams = data.get("notams", [])
                if notams:
                    for i, notam in enumerate(notams[:10]):
                        title = notam.get("notam_id", f"NOTAM #{i+1}")
                        with st.expander(f"⚠️ {title}"):
                            st.code(notam.get("raw_text", str(notam)), language="text")
                else:
                    st.success("✅ No active NOTAMs reported for this airport.")

                with st.expander("View Full JSON Telemetry"):
                    st.json(data)
