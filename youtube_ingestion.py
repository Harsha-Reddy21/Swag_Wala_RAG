import os
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_community.document_loaders import YoutubeLoader
from langchain.schema import Document
from fastembed import TextEmbedding
from qdrant_client import QdrantClient, models
from typing import List
import uuid

# Load environment variables
load_dotenv()



class YouTubeIngestion:
    def __init__(self, embedding_model="nomic-ai/nomic-embed-text-v1", collection_name="youtube-transcripts"):
        """
        Initialize the YouTube ingestion pipeline.
        
        Args:
            embedding_model (str): FastEmbed model name to use for embeddings.
            collection_name (str): Name of the Qdrant collection.
        """
        # Initialize embeddings with FastEmbed
        self.model = TextEmbedding(embedding_model)
        
        # Initialize Qdrant client with cloud configuration (use env if available)
        self.client = QdrantClient(
            url=os.getenv("QDRANT_URL", "https://4fd663f5-7f29-44e3-a3ae-5c8541547802.europe-west3-0.gcp.cloud.qdrant.io:6333"),
            api_key=os.getenv("QDRANT_API_KEY", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.GKTtCzkmi8EUtKz3_EeDthvjyLDkmk0yp1YJCTcs9yQ"),
            timeout=60.0,
        )
        self.collection_name = collection_name
        
        self.chunk_size = 1000
        self.chunk_overlap = 200
        
        # Create collection if it doesn't exist
        self._create_collection()

    def _format_mm_ss(self, seconds: float) -> str:
        """
        Convert seconds (float) to mm:ss string.
        """
        total_seconds = int(seconds)
        minutes = total_seconds // 60
        secs = total_seconds % 60
        return f"{minutes}:{secs:02d}"
    
    def extract_video_id(self, youtube_url):
        """
        Extract video ID from YouTube URL.
        
        Args:
            youtube_url (str): YouTube URL
            
        Returns:
            str: Video ID
        """
        if "youtube.com/watch?v=" in youtube_url:
            return youtube_url.split("v=")[1].split("&")[0]
        elif "youtu.be/" in youtube_url:
            return youtube_url.split("/")[-1].split("?")[0]
        else:
            # Assume it's already a video ID
            return youtube_url
    
    def get_transcript(self, video_id):
        """
        Get transcript from YouTube video.

        Args:
            video_id (str): YouTube video ID

        Returns:
            list[dict]: Transcript segments with timing
        """
        try:
            # Create API instance and fetch transcript
            ytt_api = YouTubeTranscriptApi()
            fetched_transcript = ytt_api.fetch(video_id, languages=['en'])
            
            # Convert to raw data (list of dictionaries)
            transcript_list = fetched_transcript.to_raw_data()
            
            total_chars = sum(len(item.get('text', '')) + 1 for item in transcript_list)
            print(f"Successfully extracted transcript. Segments: {len(transcript_list)}, Total length: {total_chars} characters")
            return transcript_list
            
        except Exception as e:
            print(f"Error getting transcript: {e}")
            raise
    
    def create_chunks(self, transcript_segments: List[dict]):
        """
        Split transcript into chunks while preserving start/end times.
        
        Args:
            transcript_segments (list[dict]): Items with 'text','start','duration'
            
        Returns:
            list: List of Document objects with timing metadata
        """
        chunks: List[Document] = []
        n = len(transcript_segments)
        i = 0
        target_size = self.chunk_size
        overlap_chars = self.chunk_overlap

        while i < n:
            j = i
            current_chars = 0
            while j < n and current_chars < target_size:
                current_chars += len(transcript_segments[j].get('text', '')) + 1
                j += 1

            if j == i:
                j = min(i + 1, n)

            segment_slice = transcript_segments[i:j]
            chunk_text = " ".join(seg.get('text', '') for seg in segment_slice)
            start_time = float(segment_slice[0].get('start', 0.0))
            last = segment_slice[-1]
            end_time = float(last.get('start', 0.0)) + float(last.get('duration', 0.0))

            chunks.append(
                Document(
                    page_content=chunk_text,
                    metadata={
                        "start_time": start_time,
                        "end_time": end_time,
                        "start_time_str": self._format_mm_ss(start_time),
                        "end_time_str": self._format_mm_ss(end_time),
                    },
                )
            )

            if j >= n:
                break

            # compute next i using character overlap backwards from j
            remaining_overlap = overlap_chars
            k = j - 1
            while k > i and remaining_overlap > 0:
                remaining_overlap -= len(transcript_segments[k].get('text', '')) + 1
                k -= 1
            i = max(i, k + 1)

        print(f"Created {len(chunks)} chunks")
        print(chunks[0])
        print(chunks[0].metadata)
        return chunks
    
    def _create_collection(self):
        """
        Create Qdrant collection if it doesn't exist. Idempotent and tolerant to timeouts.
        """
        try:
            self.client.get_collection(self.collection_name)
            print(f"Collection '{self.collection_name}' already exists")
            return
        except Exception as e:
            print(f"get_collection failed ({e}); attempting to create collection...")

        try:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=768,  # nomic-embed-text-v1 produces 768-dimensional vectors
                    distance=models.Distance.COSINE,
                ),
            )
            print(f"Created collection '{self.collection_name}'")
        except Exception as create_exc:
            message = str(create_exc).lower()
            if "already exists" in message or "409" in message or "conflict" in message:
                print(f"Collection '{self.collection_name}' already exists; proceeding.")
            else:
                raise
    
    def create_vector_store(self, chunks):
        """
        Create Qdrant vector store from chunks.
        
        Args:
            chunks (list): List of Document objects
            
        Returns:
            int: Number of documents stored
        """
        print("Creating embeddings and storing in Qdrant...")
        
        # Prepare texts for embedding
        texts = [chunk.page_content for chunk in chunks]
        
        # Generate embeddings
        embeddings = list(self.model.embed(texts))
        
        # Prepare points for Qdrant
        points = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            point_id = str(uuid.uuid4())
            points.append(
                models.PointStruct(
                    id=point_id,
                    vector=embedding.tolist(),
                    payload={
                        "text": chunk.page_content,
                        "metadata": chunk.metadata
                    }
                )
            )
        
        # Upsert points to Qdrant
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )
        
        print(f"Stored {len(chunks)} documents in Qdrant collection '{self.collection_name}'")
        return len(chunks)
    
    def similarity_search(self, query, k=3):
        """
        Perform similarity search using Qdrant.
        
        Args:
            query (str): Search query
            k (int): Number of results to return
            
        Returns:
            list: List of search results
        """
        # Generate embedding for query
        query_embedding = next(self.model.embed([query]))
        
        # Search in Qdrant
        search_results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding.tolist(),
            limit=k
        )
        
        # Convert results to Document-like format
        results = []
        for result in search_results:
            doc = Document(
                page_content=result.payload["text"],
                metadata=result.payload["metadata"]
            )
            results.append(doc)
        
        return results
    
    def ingest_youtube_video(self, youtube_url):
        """
        Complete ingestion pipeline for a YouTube video.
        
        Args:
            youtube_url (str): YouTube URL or video ID
            
        Returns:
            int: Number of documents stored
        """
        print(f"Starting ingestion for: {youtube_url}")
        
        # Extract video ID
        video_id = self.extract_video_id(youtube_url)
        print(f"Video ID: {video_id}")
        
        # Get transcript segments (with timings)
        transcript_segments = self.get_transcript(video_id)

        # Create chunks with start/end times
        chunks = self.create_chunks(transcript_segments)
        
        # Create vector store
        num_docs = self.create_vector_store(chunks)
        
        print("Ingestion completed successfully!")
        return num_docs


def main():
    """
    Example usage of the YouTube ingestion pipeline.
    """
    # Initialize ingestion pipeline
    ingestion = YouTubeIngestion()
    
    # Example video ID from the notebook
    video_url = "https://www.youtube.com/watch?v=w7sai6uWH3o"  # or full URL: "https://www.youtube.com/watch?v=Gfr50f6ZBvo"
    
    # Ingest the video
    num_docs = ingestion.ingest_youtube_video(youtube_url=video_url)
    
    print(f"\nVector store info:")
    print(f"Number of documents: {num_docs}")
    
    # Example: Test similarity search
    query = "What was mentioned in Bhagavth Gita"
    results = ingestion.similarity_search(query, k=3)
    print(f"\nSimilarity search results for '{query}':")
    for i, result in enumerate(results, 1):
        print(f"\nResult {i}:")
        print(result.page_content[:200] + "...")


if __name__ == "__main__":
    main()