import requests
import re
import math
from datetime import datetime

class WeatherParser:
    def __init__(self):
        self.noaa_endpoint = "https://aviationweather.gov/api/data/metar"
        self.airport_endpoint = "https://aviationweather.gov/api/data/airport"
        self.notam_endpoint = "https://aviationweather.gov/api/data/notam"
        self.airsigmet_endpoint = "https://aviationweather.gov/api/data/airsigmet"
        self.ollama_url = "http://10.11.12.60:11434/api/generate"
        self.model_name = "llama3.1:8b"
        self.runway_cache = {}

    def _extract_icao(self, raw_metar: str) -> str:
        tokens = raw_metar.strip().split()
        for token in tokens:
            if re.match(r'^[A-Z]{4}$', token):
                return token
        return tokens[0] if tokens else "UNKNOWN"

    def _normalize_runway_ids(self, runways):
        normalized = []
        for r in runways:
            if isinstance(r, dict) and 'id' in r:
                rid = r['id']
            elif isinstance(r, str):
                rid = r
            else:
                continue
            if '/' in rid:
                normalized.extend([x.strip() for x in rid.split('/')])
            else:
                normalized.append(rid.strip())
        return [x for x in normalized if x]

    def fetch_live_metar(self, airport_icao: str) -> tuple[str, bool]:
        icao_clean = airport_icao.strip().upper()
        params = {"ids": icao_clean, "format": "raw"}
        try:
            response = requests.get(self.noaa_endpoint, params=params, timeout=10)
            if response.status_code == 200 and response.text.strip():
                raw_text = response.text.strip()
                if len(raw_text) >= 5 and not raw_text.startswith("ERROR"):
                    return raw_text, True
            params["hours"] = "3"
            response = requests.get(self.noaa_endpoint, params=params, timeout=10)
            if response.status_code == 200 and response.text.strip():
                raw_text = response.text.strip()
                if len(raw_text) >= 5:
                    return raw_text, False
            return f"ERROR: No METAR data available for {icao_clean} in the last 3 hours.", False
        except Exception as e:
            return f"ERROR: NOAA Network issue: {str(e)}", False

    def fetch_airport_info(self, icao: str) -> dict:
        icao = icao.upper().strip()
        if icao in self.runway_cache:
            return self.runway_cache[icao]

        try:
            params = {"ids": icao, "format": "json"}
            response = requests.get(self.airport_endpoint, params=params, timeout=12)
            if response.status_code == 200:
                data = response.json()
                if isinstance(data, list) and len(data) > 0:
                    airport_info = data[0]
                    self.runway_cache[icao] = airport_info
                    return airport_info
        except:
            pass
        return {"runways": []}

    def get_runways_for_airport(self, raw_metar: str) -> list:
        icao = self._extract_icao(raw_metar)
        airport_info = self.fetch_airport_info(icao)
        raw_runways = airport_info.get("runways", [])
        return self._normalize_runway_ids(raw_runways)

    def calculate_runway_wind(self, wind_dir: int, wind_speed: int, runway_id: str) -> dict:
        try:
            num_match = re.search(r'(\d{1,2})', runway_id)
            if not num_match:
                return {"direction": runway_id, "headwind": 0, "crosswind": 0, "is_best": False}
            
            rwy_heading = int(num_match.group(1)) * 10
            diff = min(abs(wind_dir - rwy_heading), 360 - abs(wind_dir - rwy_heading))
            headwind = round(wind_speed * math.cos(math.radians(diff)))
            crosswind = round(abs(wind_speed * math.sin(math.radians(diff))))
            
            return {
                "direction": runway_id,
                "headwind": max(0, headwind),
                "crosswind": crosswind,
                "is_best": False
            }
        except:
            return {"direction": runway_id, "headwind": 0, "crosswind": 0, "is_best": False}

    def fetch_airsigmets(self, icao: str) -> list:
        try:
            params = {"ids": icao.upper(), "format": "json"}
            response = requests.get(self.airsigmet_endpoint, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if isinstance(data, list):
                    return data
        except Exception:
            pass
        return []

    def decode_airsigmet(self, raw_text: str) -> list:
        icao = self._extract_icao(raw_text)
        return self.fetch_airsigmets(icao)

    def decode_airsigmet_plain_english(self, sigmets: list) -> str:
        if not sigmets:
            return "No active AIRMETs or SIGMETs."

        lines = ["🌩️ **Active Convective SIGMETs:**\n"]
        
        for sig in sigmets:
            series = sig.get('seriesId', 'Unknown')
            tops = sig.get('altitudeHi1', 'Unknown')
            dir = sig.get('movementDir')
            spd = sig.get('movementSpd')
            movement = f" moving from {dir}° at {spd} kt" if dir and spd else " (stationary or slow)"

            lines.append(f"**SIGMET {series}**")
            lines.append(f"• Severe Thunderstorms")
            lines.append(f"• Tops up to {tops:,} ft")
            lines.append(f"•{movement}")
            lines.append(f"• Valid until {sig.get('validTimeTo', 'Unknown')}")
            lines.append("")

        return "\n".join(lines)

    def decode_metar_regex(self, raw_metar: str) -> dict:
        clean_text = raw_metar.strip().replace('$', '').replace('"', '')
       
        output = {
            "airport": self._extract_icao(raw_metar),
            "time": "",
            "wind_dir": 0,
            "wind_speed_kt": 0,
            "wind_speed_mph": 0,
            "wind_gust_kt": 0,
            "wind_gust_mph": 0,
            "visibility": "10SM",
            "clouds": "Clear Sky",
            "ceiling_ft": 99999,
            "temp_c": None,
            "dew_c": None,
            "temp_f": None,
            "dew_f": None,
            "altimeter": 29.92,
            "slp_hpa": None,
            "pressure_tendency": "",
            "precip": "No measurable precipitation reported",
            "flight_rules": "VFR",
            "raw_feed": clean_text,
            "status": "Success",
            "remarks": ""
        }
        
        time_match = re.search(r'\b(\d{6}Z)\b', clean_text)
        if time_match:
            output["time"] = time_match.group(1)

        wind_match = re.search(r'(\d{3}|VRB)(\d{2,3})(G(\d{2,3}))?KT', clean_text)
        if wind_match:
            dir_text = wind_match.group(1)
            output["wind_dir"] = 0 if dir_text == "VRB" else int(dir_text)
            output["wind_speed_kt"] = int(wind_match.group(2))
            output["wind_speed_mph"] = round(output["wind_speed_kt"] * 1.15078)
            if wind_match.group(4):
                output["wind_gust_kt"] = int(wind_match.group(4))
                output["wind_gust_mph"] = round(output["wind_gust_kt"] * 1.15078)

        vis_match = re.search(r'(\d+/?\d*)SM', clean_text)
        if vis_match:
            output["visibility"] = vis_match.group(0)

        alt_match = re.search(r'A(\d{4})', clean_text)
        if alt_match:
            val = alt_match.group(1)
            output["altimeter"] = float(f"{val[:2]}.{val[2:]}")

        temp_match = re.search(r'\b(M?\d{2})/(M?\d{2})\b', clean_text)
        if temp_match:
            t_txt = temp_match.group(1).replace('M', '-')
            d_txt = temp_match.group(2).replace('M', '-')
            output["temp_c"] = int(t_txt)
            output["dew_c"] = int(d_txt)
            output["temp_f"] = round(output["temp_c"] * 9/5 + 32)
            output["dew_f"] = round(output["dew_c"] * 9/5 + 32)

        t_group = re.search(r'T(\d{4})(\d{4})', clean_text)
        if t_group:
            t1 = int(t_group.group(1))
            t2 = int(t_group.group(2))
            output["temp_c"] = t1 / 10.0 if t1 < 1000 else -(t1 - 1000) / 10.0
            output["dew_c"] = t2 / 10.0 if t2 < 1000 else -(t2 - 1000) / 10.0
            output["temp_f"] = round(output["temp_c"] * 9/5 + 32)
            output["dew_f"] = round(output["dew_c"] * 9/5 + 32)

        cloud_layers = re.findall(r'(FEW|SCT|BKN|OVC|VV)(\d{3})', clean_text)
        if cloud_layers:
            output["clouds"] = " ".join([f"{c[0]}{c[1]}" for c in cloud_layers])
            for layer, height_code in cloud_layers:
                if layer in ["BKN", "OVC", "VV"]:
                    height_ft = int(height_code) * 100
                    if height_ft < output["ceiling_ft"]:
                        output["ceiling_ft"] = height_ft

        slp_match = re.search(r'SLP(\d{3})', clean_text)
        if slp_match:
            slp = int(slp_match.group(1))
            output["slp_hpa"] = 1000 + slp / 10.0 if slp < 500 else 900 + slp / 10.0

        p_tend = re.search(r'5(\d)(\d{3})', clean_text)
        if p_tend:
            char = p_tend.group(1)
            change = int(p_tend.group(2)) / 10.0
            tend_desc = {
                '0': 'Increasing then decreasing', '1': 'Increasing steadily or unsteadily',
                '2': 'Increasing then steady', '3': 'Decreasing then increasing',
                '4': 'Decreasing steadily or unsteadily', '5': 'Decreasing then steady',
                '6': 'Steady or increasing then decreasing', '7': 'Steady then decreasing',
                '8': 'Steady',
            }.get(char, f'Code {char}')
            output["pressure_tendency"] = f"{tend_desc}, change of {change:.1f} hPa over last 3 hours"

        precip_match = re.search(r'(?:RMK| )6(\d{4})', clean_text)
        if precip_match:
            amount = int(precip_match.group(1))
            if amount == 0:
                output["precip"] = "No measurable precipitation in past 6 hours"
            else:
                inches = amount / 100.0
                output["precip"] = f"{inches:.2f} inches in past 6 hours"

        vis_num = 10.0
        vis_digits = "".join(c for c in output["visibility"] if c.isdigit() or c == '.')
        if vis_digits:
            try: vis_num = float(vis_digits)
            except: pass
                
        if output["ceiling_ft"] < 1000 or vis_num < 3.0:
            output["flight_rules"] = "IFR"
        elif output["ceiling_ft"] <= 3000 or vis_num <= 5.0:
            output["flight_rules"] = "MVFR"

        return output

    def fetch_ai_briefing(self, raw_metar: str, decoded: dict = None) -> str:
        if decoded is None:
            decoded = self.decode_metar_regex(raw_metar)
        
        current_time = datetime.now().strftime("%H:%M")
        hour = int(current_time.split(':')[0])
        greeting = "Good afternoon" if 12 <= hour < 17 else "Good evening" if hour >= 17 else "Good morning"
        
        prompt = f"""You are giving a professional, concise pilot briefing. {greeting}, pilots. Current time is {current_time}.

METAR: {raw_metar}

Key decoded info:
- Wind: {decoded.get('wind_dir', 0)}° at {decoded.get('wind_speed_kt')} knots ({decoded.get('wind_speed_mph')} mph)
- Visibility: {decoded.get('visibility')}
- Clouds: {decoded.get('clouds')}
- Ceiling: {decoded.get('ceiling_ft')} ft AGL
- Temp: {decoded.get('temp_c')}°C ({decoded.get('temp_f')}°F)
- Dew point: {decoded.get('dew_c')}°C ({decoded.get('dew_f')}°F)
- Altimeter: {decoded.get('altimeter')} inHg
- Flight Rules: {decoded.get('flight_rules', 'VFR')}

Create a **single, natural, concise paragraph** pilot briefing. 
Start with wind, then visibility and sky conditions (include ceiling), mention temperature if relevant, altimeter, and any notable hazards or trends. 
Use real pilot language. Be honest about the conditions (especially if IFR). Do not use informal slang like "VFR temperature"."""

        payload = {"model": self.model_name, "prompt": prompt, "stream": False, "temperature": 0.3}
        
        try:
            res = requests.post(self.ollama_url, json=payload, timeout=25)
            return res.json().get("response", "Briefing unavailable.").strip()
        except Exception as e:
            return f"Local AI unavailable: {str(e)}"

    def decode_metar(self, raw_metar: str) -> dict:
        try:
            data = self.decode_metar_regex(raw_metar)
            icao = data["airport"]
            
            runways = self.get_runways_for_airport(raw_metar)
            wind_dir = data.get("wind_dir", 0)
            wind_speed = data.get("wind_speed_kt", 0)
            
            runway_report = [self.calculate_runway_wind(wind_dir, wind_speed, rwy_id) for rwy_id in runways]
            
            if runway_report:
                best = max(runway_report, key=lambda x: x["headwind"])
                best["is_best"] = True
            runway_report.sort(key=lambda x: x["headwind"], reverse=True)
            
            data["airmets_sigmets"] = self.fetch_airsigmets(icao)
            data["runway_report"] = runway_report
            data["notams"] = []
            data["ai_briefing"] = self.fetch_ai_briefing(raw_metar, data)
            return data
        except Exception as e:
            return {"status": "Error", "message": str(e), "raw_feed": raw_metar}

    def generate_detailed_briefing(self, decoded: dict) -> str:
        wind_dir = decoded.get("wind_dir", 0)
        wind_dir_str = f"From {wind_dir}°" if wind_dir else "Variable"
        
        return f"""**METAR Summary for {decoded['airport']}**
Time: {decoded.get('time', 'N/A')}

**Wind & Visibility**
Wind: {wind_dir_str} at {decoded.get('wind_speed_kt')} knots ({decoded.get('wind_speed_mph')} mph)
Visibility: {decoded.get('visibility', 'N/A')}

**Sky Conditions**
{decoded.get('clouds', 'Clear')}
Ceiling: {decoded.get('ceiling_ft', 'Unlimited')} ft AGL

**Atmosphere**
Temperature: {decoded.get('temp_c')}°C ({decoded.get('temp_f')}°F)
Dew point: {decoded.get('dew_c')}°C ({decoded.get('dew_f')}°F)
Altimeter: {decoded.get('altimeter', 'N/A')} inHg
Sea Level Pressure: {decoded.get('slp_hpa', 'N/A')} hPa
Pressure Trend: {decoded.get('pressure_tendency', 'N/A')}
Precipitation: {decoded.get('precip')}

**Flight Rules**: {decoded.get('flight_rules', 'VFR')}
"""