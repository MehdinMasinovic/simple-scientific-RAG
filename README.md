# Simple scientific RAG

## Requirements

Make sure you have Git, Python 3.12+ and Docker installed on your machine.

## Setup Instructions

1. **Clone the repository**

    ```bash
    git clone git@github.com:lebe1/simple-scientific-RAG.git
    cd simple-scientific-RAG
    ```

2. **Create a virtual environment** (optional but recommended)

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. **Install dependencies**

    Before running the project, install all the required dependencies using `pip`:

    ```bash
    pip install -r requirements.txt
    ```

4. **Build Dockerfile**

    Copy the example environment file into the `env`

    ```bash
    cp .env.example .env
    ```

    ```bash
    docker compose build
    ```

5. **Run docker container**

   Run docker-compose.yml to pull the required image:
   ```bash
   docker compose up -d
   ```
6. **Install Ollama model llama3.2**

   Pull the required model llama3.2 by running:

   ```bash
   docker exec ollama ollama run llama3-chatqa:8b
   docker exec ollama ollama run gemma3:12b   
   ```

7. **Install model for chunking**

    ```bash
    python -m spacy download de_core_news_lg
    ```

8. **Create the index from the legal text**

    ```bash
    python app/workflow.py update-es-index
    ```

## Running the application after setup instructions

If Docker containers are not running yet, start them again with:

```bash
docker compose up -d
```

**Note:** If you want to improve your runtime and you have access to a GPU, comment out the commented lines of code in the `docker-compose.yml`

Run fastapi server locally:

```bash
cd app;
uvicorn main:app --reload
```

### Testing the API

There are two ways for testing the API.  
Either by sending the following POST-request using `curl`:
```bash
curl -X POST "http://127.0.0.1:8000/api/rag" -H "Content-Type: application/json" -d '{"question": "Wie hoch darf ein Gebäude in Bauklasse I gemäß Artikel IV in Wien sein?", "model":"jinaai/jina-embeddings-v2-base-de", "spacy_model":"de_core_news_lg", "chunk_size_in_kb":4}'
```
```bash
curl -X POST "http://127.0.0.1:8000/api/search" -H "Content-Type: application/json" -d '{"query": "Wie hoch darf ein Gebäude in Bauklasse I gemäß Artikel IV in Wien sein?", "model":"jinaai/jina-embeddings-v2-base-de", "spacy_model":"de_core_news_lg", "chunk_size_in_kb":4}'
```

Or by opening the built-in Swagger of FastAPI via `http://127.0.0.1:8000/docs`

### Recreating the embeddings

In case the current datafile `data/legal-basis.txt` is changed or extended, you have to re-create the embeddings as follows:

```bash
# This assumes you have the venv environment enabled and you are inside the app/ directory
python workflow.py create-embeddings
```

It takes the datafile `data/legal-basis.txt` as input, chunks it into 4kb parts, and computes the embeddings using `jinaai/jina-embeddings-v2-base-de`.  
Afterwards, the results of the current datafile are stored in the numpy array `data/embeddings.npy` to later on load it easily. We copied the `data/embeddings.npy` to `data/jina-embeddings-v2-base-de-4kb` to store the embeddings created from 4kb chunks as backup/reference. 

## Running the automated question query

For now, this only works when the fastapi server is called outside of docker. If the fastapi server is running on docker, you need to stop it first, to be able to execute the following command:

```bash
cd app
```

```bash
uvicorn main:app --reload
```

Open a second terminal in the same directory and run:

```bash
python question_query.py
```

## Data
The data of the legal basis can be found under https://www.ris.bka.gv.at/GeltendeFassung.wxe?Abfrage=LrW&Gesetzesnummer=20000006

# Fancy TODOs
- Store everything in the database to use variables in RAM as less as possible
- Mount data directory in fastapi server container
- Create API call to trigger question_query.py
- Clean up starting processes into one single docker-compose.yml
- Work with logs instead of prints
- Use try catch phrases for better error detection
- Add CI for linting
- Add tests?