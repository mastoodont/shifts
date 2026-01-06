# Shift Scheduler Backend

## Local Development Setup

### Prerequisites

- Python 3.8+
- pip

### Setup Instructions

1. **Create a virtual environment:**

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

2. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment variables:**

   ```bash
   # Copy the example environment file
   cp .env.example .env
   
   # Edit .env and fill in your credentials:
   # - PAYPAL_CLIENT: Your PayPal client ID
   # - PAYPAL_SECRET: Your PayPal secret key
   # - DATABASE_FILE: (optional) Path to SQLite database file (defaults to 'shift_backend.db')
   ```

4. **Run the application:**

   ```bash
   # From the repository root:
   uvicorn backend.main:app --reload
   
   # Or from the backend directory:
   cd backend
   uvicorn main:app --reload
   ```

   The API will be available at `http://localhost:8000`

5. **Check application health:**

   ```bash
   curl http://localhost:8000/health
   ```

   Expected response: `{"status":"ok","db":true}`

### API Documentation

Once running, visit `http://localhost:8000/docs` for interactive API documentation (Swagger UI).

### Database

The application uses SQLite by default. The database file will be created automatically on first run. You can customize the database file location using the `DATABASE_FILE` environment variable in your `.env` file.

### Important Notes

- **Never commit** your `.env` file or any files containing real secrets to the repository
- The `.env.example` file is provided as a template only
- For production deployment, ensure proper secret management practices
