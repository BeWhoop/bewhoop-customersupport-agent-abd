import os
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()

# Initialize Supabase client
supabase_url = os.environ.get("SUPABASE_URL")
supabase_key = os.environ.get("SUPABASE_KEY")
supabase_client = create_client(supabase_url, supabase_key)

# Test connection
try:
    response = supabase_client.table('documents').select("*").limit(1).execute()
    if response:
        print("Supabase connection successful!")
except Exception as e:
    print("Supabase connection failed:", e)