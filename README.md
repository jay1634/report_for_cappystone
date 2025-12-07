# Project Module Analysis & Architecture Report

## 1. Custom Modules Created (Internal Codebase)
This section lists every source file created for this project and explains its role.

### Backend Core (`backend/`)

#### `main.py`
*   **Purpose**: The central API server using FastAPI. It defines endpoints (`/chat`, `/generate_itinerary`, `/routes`) and orchestrates the flow between the user, LLM, RAG, and tools.
*   **Why used**: To serve the application as a REST API, allowing decoupling between the python backend and any frontend (web, mobile).
*   **Best at this place**: Acts as the gatekeeper. It forces specific logic (like injecting weather data into prompts) before the LLM ever sees the user's message.

#### `memory.py`
*   **Purpose**: Manages persistent storage using SQLite. Handles `chat_history` (past messages) and `user_prefs` (user name, budget, interests).
*   **Why used**: To maintain conversational context over time.
*   **Best at this place (SQLite vs Redis)**:
    *   **SQLite** is used here because it is serverless and zero-conf. It stores data in a single file (`memory.db`), making the project portable.
    *   **Vs Redis**: Redis is an in-memory store requiring a separate server process. For a single-user or small-scale travel planner, Redis introduces unnecessary deployment complexity (Docker requirements) without offering significant benefits over SQLite, which is fast enough for text chat history.

#### `config.py`
*   **Purpose**: Centralizes configuration variables (API keys, file paths, model names).
*   **Why used**: To avoid hardcoding sensitive keys (Groq, OpenWeather) or paths in multiple files.
*   **Best at this place**: changing a key or model name here updates it globally across the app.

#### `guardrails.py`
*   **Purpose**: A keyword-based safety filter to block processing of unsafe or irrelevant topics (e.g., violence, illegal acts).
*   **Why used**: To Ensure the agent remains focused on travel and doesn't produce harmful content.
*   **Best at this place**: A simple, fast pre-check before calling expensive LLM or API services.

#### `llm_client.py`
*   **Purpose**: A wrapper around the Groq API client `ChatGroq`.
*   **Why used**: To handle the direct connection to the Large Language Model.
*   **Best at this place**: Abstracts the model details. If the underlying model changes (e.g., Llama-3 to Mixtral), changes are confined to this file.

#### `langchain_agent.py`
*   **Purpose**: Defines a sophisticated LangChain agent with tool-calling capabilities (`create_tool_calling_agent`).
*   **Why used**: Provides a more autonomous agent that can decide *which* tool to call (Weather vs Routes) based on user input, rather than hardcoded logic.
*   **Best at this place**: Separates complex agentic logic from the simple request-response logic in `main.py`. (Note: This appears to be an advanced alternative to the simpler logic currently used in `main.py`).

### Data & RAG (`backend/`)

#### `rag_pipeline.py`
*   **Purpose**: Implements the retrieval logic. Loads the vector index and finds relevant text chunks for a query.
*   **Why used**: To provide the "Knowledge" of the system, grounding answers in the provided travel documents.
*   **Best at this place**: Encapsulates the math/logic of TF-IDF and cosine similarity.

#### `build_vector_store.py`
*   **Purpose**: A simplified script to ingest text files from `data/corpus`, chunk them, and build the FAISS vector index.
*   **Why used**: "Offline" processing. This runs once to prepare data, so the main app doesn't have to rebuild the index every restart.
*   **Best at this place**: Keeps heavy data processing out of the runtime request loop.

### Domain Logic (`backend/`)

#### `itinerary.py`
*   **Purpose**: Specialized logic for generating day-by-day travel plans.
*   **Why used**: Itinerary generation requires specific, large prompts and structured data parsing different from normal chat.
*   **Best at this place**: Isolates the "Planner" capability from the "Chat" capability.

### Tools (`backend/tools/`)

#### `weather_tool.py`
*   **Purpose**: Fetches real-time weather from OpenWeatherMap.
*   **Why used**: LLMs have a knowledge cutoff; they don't know today's weather.
*   **Best at this place**: Modular tool that can be plugged into any agent.

#### `free_routes_tool.py`
*   **Purpose**: Fetches routing data (distance/duration) between cities.
*   **Why used**: To give realistic travel estimates.
*   **Best at this place**: Handles the specific API format of the routing provider.

#### `osrm_routes_tool.py`
*   **Purpose**: An alternative routing tool using OSRM (Open Source Routing Machine).
*   **Why used**: Provides a backup or alternative source for routing data if the primary one fails or isn't preferred.
*   **Best at this place**: keeps specific OSRM logic separate from generic routing logic.

### Frontend (`frontend/`)

#### `app.py`
*   **Purpose**: The main user interface build with Streamlit.
*   **Why used**: To render chat bubbles, input forms, and maps in the browser.
*   **Best at this place**: Streamlit allows building the UI entirely in Python, syncing perfectly with the backend data structures.

#### `api_client.py`
*   **Purpose**: Handles HTTP requests from the Frontend to the Backend.
*   **Why used**: Decouples the UI code from the network logic. `app.py` calls `api_chat`, which handles the `requests.post` complexity.
*   **Best at this place**: If the backend URL changes (e.g., to a cloud server), only this file needs to be updated.

---

## 2. External Modules & Libraries (Dependencies)
Why these specific libraries were chosen over others.

#### **FastAPI** (`fastapi`)
*   **Used for**: Backend API.
*   **Why**: Async-native, high performance, auto-validation. **Best vs Flask/Django** for modern AI microservices due to speed and developer experience (Swagger UI).

#### **Groq** (`groq`)
*   **Used for**: LLM Inference.
*   **Why**: **Speed**. Groq LPU chips deliver tokens hundreds of times faster than standard GPUs. **Best vs OpenAI/Ollama** for real-time chat where latency measures in milliseconds matter.

#### **FAISS** (`faiss-cpu`)
*   **Used for**: Vector Search.
*   **Why**: efficient similarity search. **Best vs Chroma/Pinecone** for this project because it's a lightweight library (no server needed) that handles the scale of travel guides perfectly.

#### **LangChain** (`langchain`, `langchain_groq`)
*   **Used for**: Agent framework and document splitting.
*   **Why**: Standard abstraction for chaining LLM calls and tools. **Best vs writing raw prompts** because it handles history management and tool binding automatically.

#### **Streamlit** (`streamlit`, `streamlit-folium`)
*   **Used for**: Frontend.
*   **Why**: Rapid prototyping. **Best vs React/Vue** for a single-developer or data-focused project where the goal is functionality over custom CSS pixel-perfection.

#### **Polyline** (`polyline`)
*   **Used for**: Decoding route geometries.
*   **Why**: Maps APIs often return compressed "polyline" strings to save bandwidth. This library decodes them into GPS coordinates for drawing on the map.

#### **PyDantic** (`pydantic`)
*   **Used for**: Data validation.
*   **Why**: Ensures that data entering the system (e.g., "days" must be an integer) is valid before processing.
