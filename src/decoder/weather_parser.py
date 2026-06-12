import requests
import re
import math

class WeatherParser:
    def __init__(self):
        self.noaa_endpoint = "https://aviationweather.gov/api/data/metar"
        self.airport_endpoint = "https://aviationweather.gov/api/data/airport"
        self.notam_endpoint = "https://aviationweather.gov/api/data/notam"
        self.airsigmet_endpoint = "https://aviationweather.gov/api/data/airsigmet"
        self.ollama_url = "http://10.11.12.60/api/generate"
        self.model_name = "qwen2.5-coder:14b"
        self.runway_cache = {}

    def _extract_icao(self, raw_metar: str) -> str:
        tokens = raw_metar.strip().split()
        for token in tokens:
            if re.match(r'^[A-Z]{4}$', token):
                return token
        return tokens[0] if tokens else "UNKNOWN"

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
                    num = len(airport_info.get("runways", []))
                    print(f"✅ Loaded {num} real runways for {icao}")
                    return airport_info
        except Exception as e:
            print(f"❌ Airport API error for {icao}: {e}")

        return {"runways": []}

    def fetch_notams(self, icao: str) -> list:
        icao = icao.upper().strip()
        try:
            params = {"ids": icao, "format": "json"}
            response = requests.get(self.notam_endpoint, params=params, timeout=15)
            if response.status_code == 200:
                data = response.json()
                if isinstance(data, list):
                    return data[:10]
                elif isinstance(data, dict) and "notams" in data:
                    return data["notams"][:10]
            return []
        except Exception as e:
            print(f"NOTAM fetch error for {icao}: {e}")
            return []

    def fetch_airmets_sigmets(self, icao: str) -> list:
        try:
            response = requests.get(self.airsigmet_endpoint, params={"format": "json"}, timeout=15)
            if response.status_code == 200:
                data = response.json()
                if isinstance(data, list):
                    return data[:12]
            return []
        except Exception as e:
            print(f"AIRMET/SIGMET fetch error: {e}")
            return []

    def decode_airsigmet(self, raw_text: str) -> str:
        """Strict decoder to prevent hallucinations"""
        prompt = f"""You are an expert CFI. Decode this SIGMET or AIRMET into accurate plain English.

STRICT RULES:
- Only use locations and states mentioned in the text.
- "MOV FROM 22020KT" = moving FROM 220° at 20 knots. Never say 220 knots.
- Do not add extra states or locations.
- Be short and clear.

Raw Text:
{raw_text}

Plain English Summary:"""

        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "temperature": 0.0,      # Very low temperature
            "top_p": 0.7
        }
        try:
            res = requests.post(self.ollama_url, json=payload, timeout=40)
            return res.json().get("response", "Unable to decode.").strip()
        except Exception as e:
            return f"AI decoding unavailable: {str(e)}"

    def get_runways_for_airport(self, raw_metar: str) -> list:
        icao = self._extract_icao(raw_metar)
        airport_info = self.fetch_airport_info(icao)
        runways = []
        if isinstance(airport_info, dict):
            for item in airport_info.get("runways", []):
                if isinstance(item, dict) and "id" in item:
                    runways.append(item["id"])
                elif isinstance(item, str):
                    runways.append(item)
        return runways

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

    def calculate_runway_wind(self, wind_dir: int, wind_speed: int, runway_id: str) -> dict:
        numbers = re.findall(r'(\d{1,2})', runway_id)
        if not numbers:
            return {"name": runway_id, "headwind": 0.0, "crosswind": 0.0, "best_direction": runway_id}

        directions = []
        for num in numbers:
            heading = int(num) * 10
            angle_diff = math.radians(wind_dir - heading)
            headwind = wind_speed * math.cos(angle_diff)
            crosswind = abs(wind_speed * math.sin(angle_diff))
            directions.append({
                "heading": heading,
                "headwind": round(headwind, 1),
                "crosswind": round(crosswind, 1),
                "dir_name": num
            })

        best = max(directions, key=lambda x: x["headwind"])
        
        return {
            "name": f"Runway {runway_id}",
            "headwind": best["headwind"],
            "crosswind": best["crosswind"],
            "best_direction": best["dir_name"]
        }

    def decode_metar(self, raw_metar: str) -> dict:
        try:
            data = self.decode_metar_regex(raw_metar)
            runways = self.get_runways_for_airport(raw_metar)
            
            wind_dir = data.get("wind_dir", 0)
            wind_speed = data.get("wind_speed", 0)
            
            runway_report = []
            seen = set()
            for rwy_id in runways:
                if rwy_id and rwy_id not in seen:
                    seen.add(rwy_id)
                    calc = self.calculate_runway_wind(wind_dir, wind_speed, rwy_id)
                    runway_report.append(calc)
            
            runway_report.sort(key=lambda x: x["headwind"], reverse=True)
            data["runway_report"] = runway_report
            data["notams"] = self.fetch_notams(data.get("airport"))
            data["airmets_sigmets"] = self.fetch_airmets_sigmets(data.get("airport"))
            
            return data
        except Exception as e:
            return {"status": "Error", "message": str(e), "raw_feed": raw_metar}