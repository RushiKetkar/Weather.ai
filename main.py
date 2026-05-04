import json
import subprocess
import sys
import time
import os
import ollama

from strands               import Agent
from strands.models.ollama import OllamaModel
from strands.tools.mcp     import MCPClient
from mcp.client.sse        import sse_client

import rag

LANGUAGE_MODEL  = "hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF"
MCP_SERVER_URL  = "http://localhost:8000/sse"
MCP_STARTUP_SEC = 3 


def _start_mcp_server() -> subprocess.Popen:
    """Launch mcp_server.py in a subprocess and return the handle."""
    proc = subprocess.Popen(
        [sys.executable, "mcp_server.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    print(f"MCP server starting (pid {proc.pid}) …")
    time.sleep(MCP_STARTUP_SEC)
    return proc

def _pretty_banner(title: str) -> None:
    line = "─" * 60
    print(f"\n{line}\n  {title}\n{line}")

def run(city: str) -> None:

    mcp_proc = _start_mcp_server()

    try:
        model = OllamaModel(
            host=os.environ.get("OLLAMA_HOST", "http://localhost:11434"),
            model_id=LANGUAGE_MODEL,
            options={"temperature": 0.1},
        )

        mcp_client = MCPClient(lambda: sse_client(MCP_SERVER_URL))

        _pretty_banner(f"Fetching weather for: {city}")

        with mcp_client:
            tools = mcp_client.list_tools_sync()
            agent = Agent(model=model, tools=tools)

            agent_query = f"""
You are a precise data-retrieval assistant. Follow these steps exactly:

1. Call the 'get_location' tool with city="{city}".
   - If 'found' is false, reply ONLY: {{"error": "City not found", "city": "{city}"}}
   - If 'found' is true, continue to step 2.

2. Call the 'get_weather' tool with city="{city}".
   Note the values for temp, humidity, wind_speed, and dew_point.

3. Call the 'plot_weather' tool with:
     city       = "{city}"
     temp       = <float from step 2>
     humidity   = <float from step 2>
     wind_speed = <float from step 2>
     dew_point  = <float from step 2>

4. After all tool calls are complete, output ONLY this JSON on the last line
   (no extra text, no markdown fences):
   {{"city": "{city}", "temp": <float>, "humidity": <float>, "wind_speed": <float>, "dew_point": <float>}}
"""
            response   = agent(agent_query)
            raw_text   = str(response).strip()

        decoder = json.JSONDecoder()
        raw_text = raw_text.strip()
        
        obj, _ = decoder.raw_decode(raw_text)
            
        weather = obj
        
        temp       = float(weather["temp"])
        humidity   = float(weather["humidity"])
        wind_speed = float(weather["wind_speed"])
        dew_point  = float(weather["dew_point"])

        print(f"\n  Temperature : {temp} °C")
        print(f"  Humidity    : {humidity} %")
        print(f"  Wind speed  : {wind_speed} m/s")
        print(f"  Dew point   : {dew_point} °C")
        print(f"\n  Bar chart saved as: weather_{city.lower().replace(' ', '_')}.html")

        _pretty_banner("Building RAG from weather_facts.txt")
        rag.load_and_embed("weather_facts.txt")

        rag_query = (
            f"Why is the temperature {temp}°C, humidity {humidity}%, "
            f"wind speed {wind_speed} m/s, and dew point {dew_point}°C in {city}?"
        )

        _pretty_banner("Retrieving & reranking relevant facts")
        results = rag.search(rag_query, initial_k=20, final_k=5)

        print("\nTop retrieved facts:")
        for chunk, sim in results["embedding_score"].tolist():
            print(f"  [{sim:.2f}] {chunk[:110]}")

        # ── 5. LLM explanation ───────────────────────────────────────────
        _pretty_banner(f"LLM Weather Explanation — {city}")

        context = "\n".join(
            f"- {chunk}"
            for chunk, _ in results["embedding_score"].tolist()
        )

        system_prompt = f"""You are a helpful meteorology assistant.
Explain why the weather in {city} is currently as follows:
  • Temperature : {temp} °C
  • Humidity    : {humidity} %
  • Wind speed  : {wind_speed} m/s
  • Dew point   : {dew_point} °C

Use ONLY the facts below. Do not invent information.

Facts:
{context}"""

        stream = ollama.chat(
            model=LANGUAGE_MODEL,
            messages=[
                {"role": "system",  "content": system_prompt},
                {"role": "user",    "content": rag_query},
            ],
            stream=True,
        )

        for chunk in stream:
            print(chunk["message"]["content"], end="", flush=True)
        print("\n")

    finally:
        mcp_proc.terminate()
        try:
            mcp_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            mcp_proc.kill()
        print("MCP server stopped.")

if __name__ == "__main__":
    city_input = input("Enter a city name: ").strip()
    if not city_input:
        print("No city entered. Exiting.")
        sys.exit(1)
    run(city_input)
