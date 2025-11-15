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
python3 -m venv venv
source venv/bin/activate  # On macOS/Linux
pip install -r requirements.txt
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
python3 -m pytest tests/test_document_loader.py -v
python3 -m pytest tests/test_chunking_service.py -v
```
