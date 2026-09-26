import json
import threading
import time
import urllib.request

DEFAULT_LAT = 28.6139
DEFAULT_LON = 77.2090


def fetch_weather_data(lat=DEFAULT_LAT, lon=DEFAULT_LON, timeout=10):
    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        "&current=temperature_2m,cloud_cover,relative_humidity_2m"
        "&daily=cloud_cover_mean,temperature_2m_max,temperature_2m_min"
        "&past_days=7&forecast_days=1"
    )
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.loads(resp.read())


def historical_cloud_cover_avg(data):
    values = [v for v in data.get("daily", {}).get("cloud_cover_mean", []) if v is not None]
    return sum(values) / len(values) if values else None


class WeatherHistoryMonitor:
    """Fetches live + historical weather online and flags deviation from the norm."""

    def __init__(self, lat=DEFAULT_LAT, lon=DEFAULT_LON, refresh_seconds=900):
        self.lat = lat
        self.lon = lon
        self.refresh_seconds = refresh_seconds
        self._lock = threading.Lock()
        self._latest = None
        self._error = None
        self._stop = False
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        while not self._stop:
            try:
                data = fetch_weather_data(self.lat, self.lon)
                current_cloud = data.get("current", {}).get("cloud_cover")
                hist_avg = historical_cloud_cover_avg(data)
                with self._lock:
                    self._latest = {
                        "current_cloud_cover": current_cloud,
                        "historical_avg_cloud_cover": hist_avg,
                    }
                    self._error = None
            except Exception as exc:
                with self._lock:
                    self._error = str(exc)

            for _ in range(self.refresh_seconds):
                if self._stop:
                    return
                time.sleep(1)

    def get_status_text(self):
        with self._lock:
            error = self._error
            latest = self._latest

        if error:
            return f"Weather: offline ({error})"
        if not latest:
            return "Weather: fetching..."

        cur = latest["current_cloud_cover"]
        hist = latest["historical_avg_cloud_cover"]
        if cur is None or hist is None:
            return "Weather: no data"

        deviation = cur - hist
        if deviation > 10:
            tag = "ABOVE NORMAL"
        elif deviation < -10:
            tag = "BELOW NORMAL"
        else:
            tag = "NORMAL"
        return f"Cloud cover {cur:.0f}% vs 7d history {hist:.0f}% [{tag}]"

    def stop(self):
        self._stop = True
