# PartSelect Chat Agent

An AI-powered chat assistant for PartSelect e-commerce, helping customers find refrigerator and dishwasher parts, troubleshoot issues, and get installation guidance using RAG architecture with Deepseek LLM.

## Setup

### Frontend
```bash
cd Frontend
npm install
npm start
```

Frontend runs at `http://localhost:3000`

### Backend
```bash
cd backend

# create a .env file and add the following

OPENROUTER_API_KEY=your_key_here
LLM_PROVIDER=openrouter
LLM_MODEL=google/gemma-3-27b-it:free

python3 -m venv venv
source venv/bin/activate  # On macOS/Linux
pip install -r requirements.txt
### Build Vector Store (Production Data)
cd backend
source venv/bin/activate
python3 -m services.ingestion_pipeline  # Or create your own ingestion script## Testing
```

## Testing

Run backend tests:
```bash
cd backend
source venv/bin/activate
python3 -m pytest tests/ -v
```

Run specific test suites:
```bash
python3 -m tests.test_document_loader
python3 -m tests.test_chunking_service
python3 -m tests.test_embedding_service
python3 -m tests.test_ingestion_pipeline## Project Structure
```
