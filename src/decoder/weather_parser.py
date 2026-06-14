    def check_aviation_brain(self) -> tuple[bool, str]:
        """Check if Ollama / Aviation Brain is reachable"""
        payload = {
            "model": self.model_name,
            "prompt": "Say only: Aviation Brain online.",
            "stream": False,
            "temperature": 0.0
        }
        try:
            res = requests.post(self.ollama_url, json=payload, timeout=8)
            if res.status_code == 200:
                return True, "The Aviation Brain is awake. 🧠"
            else:
                return False, "The Aviation Brain is asleep. (Ollama not responding)"
        except Exception as e:
            return False, f"The Aviation Brain is asleep. (Connection error: {str(e)[:80]})"