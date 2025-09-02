import os
from typing import List

from youtube_ingestion import YouTubeIngestion
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
load_dotenv()


os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")


from langchain_openai import ChatOpenAI  # type: ignore



def retrieve(query: str, k: int = 5):
    ingestion = YouTubeIngestion()
    results = ingestion.similarity_search(query, k=k)
    
    return results


def format_context(docs: List):
    parts = []
    for i, d in enumerate(docs, 1):
        meta = d.metadata or {}
        t0 = meta.get("start_time_str", "?")
        t1 = meta.get("end_time_str", "?")
        title = meta.get("video_title") or meta.get("title") or "Unknown Video"
        url = meta.get("video_url") or ""
        
        # Clean and structure the content
        content = d.page_content.strip()
        if len(content) > 500:  # Truncate very long chunks
            content = content[:500] + "..."
        
        parts.append(f"--- EXCERPT {i} ---\nVideo: {title}\nTimestamp: {t0} - {t1}\nContent: {content}\nSource: {url}")
    
    return "\n\n".join(parts)


def answer(query: str, k: int = 5) -> str:
    docs = retrieve(query, k=k)
    context = format_context(docs)

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=(
            "You are an expert assistant analyzing YouTube video transcripts. Your task is to provide accurate, detailed answers based ONLY on the provided context.\n\n"
            "IMPORTANT RULES:\n"
            "1. Answer ONLY using information from the provided context\n"
            "2. If the context doesn't contain enough information to answer the question, say 'Based on the available context, I cannot provide a complete answer to this question.'\n"
            "3. Be specific and reference the video titles and timestamps when possible\n"
            "4. If the question is about a specific topic and it's not covered in the context, clearly state what information is missing\n"
            "5. Use the exact terminology and concepts mentioned in the transcripts\n\n"
            "Context from YouTube transcripts:\n{context}\n\n"
            "Question: {question}\n\n"
            "Answer:"
        ),
    )
    
    return llm.invoke(prompt.format(context=context, question=query)).content




if __name__ == "__main__":
    print("YouTube RAG System")
    print("=" * 50)
    
    # Test with a specific query first
    test_query = "What is Product Management?"
    result = answer(test_query, k=5)
    print(result)
  
   


