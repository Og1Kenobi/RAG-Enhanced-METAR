import requests
import re
import math

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

    def decode_metar_regex(self, raw_metar: str) -> dict:
        clean_text = raw_metar.strip().replace('$', '').replace('"', '')
        tokens = clean_text.split()
       
        output = {
            "airport": self._extract_icao(raw_metar),
            "wind_dir": 0,
            "wind_speed": 0,
            "wind_gust": 0,
            "visibility": "10SM",
            "clouds": "Clear Sky",
            "ceiling_ft": 99999,
            "temp_c": 15,
            "dew_c": 10,
            "altimeter": 29.92,
            "flight_rules": "VFR",
            "raw_feed": clean_text,
            "status": "Success"
        }
        
        wind_match = re.search(r'(\d{3}|VRB)(\d{2,3})(G(\d{2,3}))?KT', clean_text)
        if wind_match:
            dir_text = wind_match.group(1)
            output["wind_dir"] = 0 if dir_text == "VRB" else int(dir_text)
            output["wind_speed"] = int(wind_match.group(2))
            if wind_match.group(4):
                output["wind_gust"] = int(wind_match.group(4))

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

        cloud_layers = re.findall(r'(FEW|SCT|BKN|OVC|VV)(\d{3})', clean_text)
        if cloud_layers:
            output["clouds"] = " ".join([f"{c[0]}{c[1]}" for c in cloud_layers])
            for layer, height_code in cloud_layers:
                if layer in ["BKN", "OVC", "VV"]:
                    height_ft = int(height_code) * 100
                    if height_ft < output["ceiling_ft"]:
                        output["ceiling_ft"] = height_ft

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

    def fetch_ai_briefing(self, raw_metar: str) -> str:
        prompt = f"""You are an experienced CFI. Give a single clear, concise sentence briefing for a VFR pilot.
Include wind, visibility, ceiling, and hazards only.

METAR: {raw_metar}"""
        payload = {"model": self.model_name, "prompt": prompt, "stream": False, "temperature": 0.3}
        try:
            res = requests.post(self.ollama_url, json=payload, timeout=25)
            return res.json().get("response", "AI briefing unavailable.").strip()
        except Exception as e:
            return f"Local AI briefing unavailable: {str(e)}"

    def decode_metar(self, raw_metar: str) -> dict:
        try:
            data = self.decode_metar_regex(raw_metar)
            runways = self.get_runways_for_airport(raw_metar)
            
            wind_dir = data.get("wind_dir", 0)
            wind_speed = data.get("wind_speed", 0)
            
            runway_report = []
            for rwy_id in runways:
                calc = self.calculate_runway_wind(wind_dir, wind_speed, rwy_id)
                runway_report.append(calc)
            
            if runway_report:
                best = max(runway_report, key=lambda x: x["headwind"])
                best["is_best"] = True
            
            runway_report.sort(key=lambda x: x["headwind"], reverse=True)
            data["runway_report"] = runway_report
            data["notams"] = []
            data["airmets_sigmets"] = []
            return data
        except Exception as e:
            return {"status": "Error", "message": str(e), "raw_feed": raw_metar}
