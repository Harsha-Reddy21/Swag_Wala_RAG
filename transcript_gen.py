import requests
import xml.etree.ElementTree as ET

def get_transcript(video_id, lang="en"):
    url = f"https://video.google.com/timedtext?lang={lang}&v={video_id}"
    print(f"📡 Fetching transcript for video_id={video_id}, lang={lang}")
    print(f"➡️ URL: {url}")

    try:
        response = requests.get(url, timeout=10)
    except Exception as e:
        print(f"❌ Request failed: {e}")
        return None

    print(f"🔍 Status Code: {response.status_code}")
    if response.status_code != 200:
        print("❌ Non-200 status code. Transcript not available.")
        return None

    if not response.text.strip():
        print("⚠️ Empty response text. Transcript not available.")
        return None

    print(f"📄 Raw response (first 500 chars):\n{response.text[:500]}")
    
    try:
        root = ET.fromstring(response.text)
    except ET.ParseError as e:
        print(f"❌ XML Parsing Error: {e}")
        return None

    transcript = []
    for i, child in enumerate(root.findall("text")):
        start = float(child.attrib.get("start", 0))
        dur = float(child.attrib.get("dur", 0))
        text = (child.text or "").replace("\n", " ").strip()
        transcript.append({
            "start": start,
            "duration": dur,
            "text": text
        })
        if i < 5:  # show first 5 lines while debugging
            print(f"✅ Line {i+1}: start={start}, dur={dur}, text='{text}'")

    print(f"📊 Total transcript lines: {len(transcript)}")
    return transcript


# Example usage
video_id = "05rtWnCDjps"
transcript = get_transcript(video_id, "en")

if transcript:
    print("\n--- First 5 Transcript Lines ---")
    for line in transcript[:5]:
        print(line)
else:
    print("Transcript not available")
