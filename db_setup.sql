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