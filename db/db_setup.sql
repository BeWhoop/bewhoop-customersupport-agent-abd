-- Enable Vector Extension
create extension if not exists vector;

CREATE TABLE documents (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    content TEXT,
    metadata JSONB,
    embedding vector(768)  -- Adjust based on your embedding model
);

-- Vector Similarity Search Function
CREATE OR REPLACE FUNCTION match_documents(
  query_embedding vector(768),
  match_count int DEFAULT NULL,
  filter JSONB DEFAULT NULL
) RETURNS TABLE (
  id UUID,  
  content TEXT,
  metadata JSONB,
  similarity FLOAT
)
LANGUAGE plpgsql
AS $$
#variable_conflict use_column
BEGIN
  RETURN QUERY
  SELECT
    id,
    content,
    metadata,
    1 - (documents.embedding <=> query_embedding) AS similarity
  FROM documents
  WHERE 1 = 1
  ORDER BY documents.embedding <=> query_embedding
  LIMIT match_count;
END;
$$;

-- Vector Search Index with IVFFlat
CREATE INDEX ON documents USING ivfflat (embedding vector_l2_ops) WITH (lists = 100);


-- 768 dims for all-mpnet-base-v2
create table if not exists qa_memory (
  id uuid primary key default uuid_generate_v4(),
  question text not null unique,
  answer text not null,
  q_embedding vector(768) not null,
  created_at timestamptz not null default now()
);

-- Cosine index (requires normalized vectors)
create index if not exists qa_memory_q_embedding_idx
on qa_memory using ivfflat (q_embedding vector_cosine_ops) with (lists = 100);

-- RPC for semantic lookup
create or replace function match_qa_memory(
  query_embedding vector(768),
  match_threshold float,
  match_count int
)
returns table (
  id uuid,
  question text,
  answer text,
  similarity float
)
language sql stable as $$
  select
    id,
    question,
    answer,
    1 - (qa_memory.q_embedding <=> query_embedding) as similarity
  from qa_memory
  where 1 - (qa_memory.q_embedding <=> query_embedding) >= match_threshold
  order by similarity desc
  limit match_count;
$$;
