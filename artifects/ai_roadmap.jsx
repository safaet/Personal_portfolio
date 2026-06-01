import { useState } from "react";

const roadmapData = [
  {
    month: 1,
    title: "Transformer & HuggingFace Core",
    theme: "Build the foundation everyone else skips",
    color: "#00D4FF",
    accent: "#003344",
    hours: "~84 hrs total · 3 hrs/day",
    bookNote: "Ch.2: Deep learning is now a large fraction of ML — understand transformers or stay behind.",
    weeks: [
      {
        week: 1,
        goal: "Understand how transformers actually work — not just use them",
        topics: "Transformer Architecture",
        subtopics: ["Self-attention mechanism", "Multi-head attention", "Positional encoding (sinusoidal + RoPE)", "Encoder vs Decoder vs Encoder-Decoder", "LayerNorm placement pre vs post"],
        math: "Derive attention: Attention(Q,K,V) = softmax(QKᵀ/√dₖ)V. Trace dimensions for a 2-head, 4-token example on paper. Understand why √dₖ stabilizes gradients.",
        coding: "Implement bare-bones attention in PyTorch from scratch (no nn.MultiheadAttention). Input: (batch, seq, d_model). Verify output shape matches.",
        project: "Visualize attention weights for a sentence using matplotlib heatmap. Run on 3 different sentences. Note which tokens attend to which.",
        research: "Read 'Attention Is All You Need' (Vaswani et al. 2017) — sections 3 and 5 only. Note: what problem does positional encoding solve?",
        portfolio: "Create GitHub repo: 'transformer-from-scratch'. Push attention implementation with docstrings. Add a README with an attention diagram.",
        output: "Working attention module. Repo live. You can explain attention math without notes.",
        codeSnippet: `import torch
import torch.nn as nn
import math

class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k: int):
        super().__init__()
        self.scale = math.sqrt(d_k)
    
    def forward(self, Q, K, V, mask=None):
        # Q,K,V: (batch, heads, seq_len, d_k)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        weights = torch.softmax(scores, dim=-1)
        return torch.matmul(weights, V), weights`
      },
      {
        week: 2,
        goal: "Master HuggingFace ecosystem end-to-end",
        topics: "HuggingFace Transformers + Tokenizers",
        subtopics: ["AutoTokenizer internals: BPE, WordPiece, SentencePiece", "AutoModel variants: ForCausalLM, ForSequenceClassification, ForTokenClassification", "Trainer API vs manual training loop", "datasets library + streaming large corpora", "Hub: push/pull models and datasets"],
        math: "Understand softmax cross-entropy loss: L = -Σ yᵢ log(p̂ᵢ). Compute by hand for a 3-class example. Know why log-softmax + NLLLoss = CrossEntropyLoss.",
        coding: "Fine-tune distilbert-base-uncased for Bengali/English sentiment on a small dataset (500 samples). Use Trainer API. Log train/eval loss per epoch.",
        project: "Deploy the fine-tuned model as a FastAPI endpoint: POST /predict → {text, label, confidence}. Test with curl.",
        research: "HuggingFace course chapters 1-3 (free at hf.co/learn). Take notes on tokenizer vocab sizes for GPT-2 vs BERT vs LLaMA.",
        portfolio: "Push FastAPI + model to GitHub. Add sample API call in README. This is your first AI-powered API project.",
        output: "Fine-tuned sentiment model live as FastAPI endpoint. HuggingFace course Ch1-3 complete.",
        codeSnippet: `from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import Trainer, TrainingArguments
from datasets import Dataset
import torch

model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name, num_labels=2
)

def tokenize(batch):
    return tokenizer(
        batch["text"], truncation=True, 
        padding="max_length", max_length=128
    )

# datasets returns input_ids, attention_mask, labels
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    report_to="wandb",  # ← always log
)`
      },
      {
        week: 3,
        goal: "Fine-tune LLMs efficiently with LoRA — the industry standard method",
        topics: "Parameter-Efficient Fine-Tuning (PEFT) + LoRA",
        subtopics: ["LoRA math: W' = W₀ + BA where rank(B,A) = r", "QLoRA: 4-bit quantization + LoRA adapters", "Which layers to target: q_proj, v_proj, k_proj, o_proj", "PEFT library: LoraConfig, get_peft_model", "Merging adapters back into base model"],
        math: "Prove why LoRA reduces parameters: original W ∈ R^{d×k} = dk params. LoRA: B ∈ R^{d×r} + A ∈ R^{r×k} = r(d+k) params. For d=4096, k=4096, r=16: compute % reduction. Answer: ~99.6% fewer trainable params.",
        coding: "QLoRA fine-tune LLaMA-3.2-1B (or Qwen2.5-0.5B if GPU-limited) on a 1000-sample instruction dataset. Use bitsandbytes + PEFT. Track loss with wandb.",
        project: "Fine-tune on a Bengali instruction dataset from HuggingFace Hub (search 'bengali instruction'). Target: model follows Bengali instructions correctly.",
        research: "Read LoRA paper abstract + sections 1, 2, 4 (arxiv.org/abs/2106.09685). Focus on Table 2 — compare parameter counts.",
        portfolio: "Push fine-tuned adapter to HuggingFace Hub. This is your first public model. Tag it properly.",
        output: "Bengali instruction-tuned LLM adapter on HuggingFace Hub. wandb training run logged.",
        codeSnippet: `from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
import torch

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",      # NormalFloat4
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,  # QLoRA: quantize the quantization constants
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-1B",
    quantization_config=bnb_config,
    device_map="auto"
)

lora_config = LoraConfig(
    r=16,           # rank — start here, tune later
    lora_alpha=32,  # scaling: ΔW = (alpha/r) * BA
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# Expect: ~0.5% trainable — that's the LoRA efficiency`
      },
      {
        week: 4,
        goal: "Evaluate and iterate — most engineers skip this and it shows",
        topics: "LLM Evaluation + Experiment Discipline",
        subtopics: ["Perplexity: PPL = exp(H(p,q)) = exp(-1/N Σ log p(xᵢ))", "BLEU, ROUGE for generation tasks", "LLM-as-judge evaluation pattern", "Prompt sensitivity analysis", "wandb experiment comparison: track hyperparams, metrics, artifacts"],
        math: "Calculate perplexity manually for a 5-token sequence. Understand: lower PPL = model is less surprised by test data. Know: PPL is exponential of cross-entropy loss.",
        coding: "Write an evaluation harness: loop over test set, compute ROUGE-L for your Bengali fine-tuned model vs base model. Print comparison table.",
        project: "Build wandb dashboard comparing: base model vs your fine-tuned model across 3 prompts × 5 metrics. Screenshot and export as portfolio artifact.",
        research: "Read 'LMSYS Chatbot Arena: Benchmarking LLMs in the Wild' abstract. Understand Elo rating for LLM evaluation — same math as chess.",
        portfolio: "Write a 300-word GitHub README section: 'Evaluation Results' with your metrics table. Every ML project needs this. Most portfolios skip it.",
        output: "Evaluation harness written. wandb comparison dashboard. Month 1 capstone: Bengali fine-tuned LLM with evaluation report.",
        codeSnippet: `import wandb
from rouge_score import rouge_scorer
import numpy as np

wandb.init(project="bengali-llm-eval", name="lora-vs-base")

scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

def evaluate_model(model, tokenizer, test_data, model_name):
    scores = []
    for sample in test_data:
        pred = generate(model, tokenizer, sample['input'])
        score = scorer.score(sample['target'], pred)
        scores.append(score['rougeL'].fmeasure)
    
    avg = np.mean(scores)
    wandb.log({f"{model_name}/rougeL": avg})
    return avg

# Compare both models
base_score = evaluate_model(base_model, tokenizer, test_data, "base")
lora_score = evaluate_model(lora_model, tokenizer, test_data, "lora")
print(f"Improvement: {(lora_score - base_score)*100:.1f}%")
wandb.finish()`
      }
    ]
  },
  {
    month: 2,
    title: "RAG Systems + Vector Search",
    theme: "Build systems companies actually deploy",
    color: "#00FF9D",
    accent: "#003322",
    hours: "~84 hrs total · 3 hrs/day",
    bookNote: "Ch.4: Identify the business problem first. RAG solves: 'our LLM doesn't know our internal data' — a real, widespread problem.",
    weeks: [
      {
        week: 1,
        goal: "Understand embeddings mathematically — not just call an API",
        topics: "Embeddings + Semantic Search Theory",
        subtopics: ["Word2Vec: skip-gram objective, negative sampling", "Sentence-BERT: Siamese network + cosine similarity", "Embedding dimensions and their tradeoffs", "Cosine similarity vs dot product vs Euclidean distance", "OpenAI text-embedding-3 vs open models comparison"],
        math: "Derive cosine similarity: cos(θ) = (A·B)/(||A||·||B||). Prove that for unit vectors, cosine similarity = dot product. Implement both and verify they give identical results on normalized vectors.",
        coding: "Build a semantic search engine over 200 arXiv abstracts using sentence-transformers. Return top-3 most similar abstracts for any query. Measure latency.",
        project: "Semantic search over your conference paper's references. Query: 'methods similar to mine' → return 3 closest abstracts. This is directly portfolio-relevant.",
        research: "Read 'Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks' (arxiv 1908.10084) — sections 1-3 only.",
        portfolio: "Add semantic search demo to your GitHub. Create HuggingFace Space with Gradio UI: user types query, sees top-3 results with similarity scores.",
        output: "Semantic search system live on HuggingFace Spaces. You can explain embedding geometry.",
        codeSnippet: `from sentence_transformers import SentenceTransformer
import numpy as np
import faiss

model = SentenceTransformer('BAAI/bge-small-en-v1.5')  # strong open model

# Embed corpus
corpus = ["abstract 1...", "abstract 2...", ...]  # your 200 abstracts
embeddings = model.encode(corpus, normalize_embeddings=True)
# normalize_embeddings=True → unit vectors → dot product = cosine similarity
# embeddings: (200, 384) for bge-small

# FAISS index: Inner Product on normalized = cosine similarity
dim = embeddings.shape[1]  # 384
index = faiss.IndexFlatIP(dim)
index.add(embeddings.astype(np.float32))

def search(query: str, k: int = 3):
    q_emb = model.encode([query], normalize_embeddings=True)
    scores, idx = index.search(q_emb.astype(np.float32), k)
    return [(corpus[i], float(s)) for i, s in zip(idx[0], scores[0])]`
      },
      {
        week: 2,
        goal: "Master chunking strategies — the most underestimated RAG skill",
        topics: "Document Processing + Chunking Strategies",
        subtopics: ["Fixed-size vs semantic chunking", "Recursive character text splitter", "Chunk overlap and why it matters", "Parent-child document retrieval", "Metadata filtering in vector stores", "ChromaDB vs FAISS vs Weaviate tradeoffs"],
        math: "Understand recall@k and precision@k for retrieval evaluation. For a 100-doc corpus, manually compute recall@3 for 5 queries where you know ground truth answers.",
        coding: "Build a ChromaDB-backed document store. Ingest your conference paper PDF (use PyMuPDF). Implement 3 chunking strategies and compare retrieval quality.",
        project: "Create a 'Paper QA' system: upload any PDF → chunk → embed → store in ChromaDB → answer questions about it. This is a direct demo of your skills.",
        research: "Read LlamaIndex chunking documentation + 'Evaluating RAG: 5 Key Metrics' (any recent blog). Document which chunking strategy performs best on your paper.",
        portfolio: "Add Paper QA demo to HuggingFace Spaces. Users upload PDF, ask questions. Record a 60-second screen recording for LinkedIn.",
        output: "Paper QA system live. ChromaDB integration complete. Chunking comparison documented.",
        codeSnippet: `import chromadb
from langchain.text_splitter import RecursiveCharacterTextSplitter
import fitz  # PyMuPDF

def load_pdf(path: str) -> str:
    doc = fitz.open(path)
    return " ".join([page.get_text() for page in doc])

# Strategy 1: Fixed-size chunks
splitter_fixed = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=50,  # overlap prevents losing context at boundaries
    separators=["\n\n", "\n", ". ", " ", ""]
)

# Strategy 2: Semantic — split at paragraph boundaries
splitter_semantic = RecursiveCharacterTextSplitter(
    chunk_size=1024,
    chunk_overlap=100,
    separators=["\n\n", "\n"]  # prefer paragraph breaks
)

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(
    name="paper_qa",
    metadata={"hnsw:space": "cosine"}
)

def ingest(chunks: list[str], doc_id: str):
    collection.add(
        documents=chunks,
        ids=[f"{doc_id}_{i}" for i in range(len(chunks))],
        metadatas=[{"source": doc_id, "chunk_id": i} for i in range(len(chunks))]
    )`
      },
      {
        week: 3,
        goal: "Build a production-grade RAG pipeline end-to-end",
        topics: "Full RAG Pipeline Architecture",
        subtopics: ["Naive RAG vs Advanced RAG vs Modular RAG", "Query rewriting / HyDE (Hypothetical Document Embeddings)", "Reranking: cross-encoders vs bi-encoders", "Context window management for retrieved docs", "Streaming responses with FastAPI + async generators"],
        math: "Understand why reranking improves results: bi-encoder retrieves (fast, approximate) then cross-encoder reranks (slow, accurate). Cross-encoder: directly computes similarity of (query, doc) pair. Bi-encoder: independent embeddings, dot product. Tradeoff: cross-encoder is O(k×N) vs bi-encoder O(N).",
        coding: "Build full RAG: ChromaDB retrieval → cross-encoder rerank (use cross-encoder/ms-marco-MiniLM-L-6-v2) → LLM generation. Wrap in FastAPI with streaming endpoint.",
        project: "RAG over Bangladesh CSE job postings scraped from Prothom Alo jobs / LinkedIn. Query: 'which jobs need PyTorch?' → retrieves relevant postings. Practical + unique.",
        research: "Read 'Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks' (Lewis et al. 2020, arxiv 2005.11401) — sections 1-3. This is the foundational RAG paper.",
        portfolio: "Deploy RAG API to Hugging Face Spaces or Railway. Document API endpoints. Add architecture diagram to README using draw.io or excalidraw.",
        output: "Production RAG API deployed. Architecture documented. You can explain every component's purpose.",
        codeSnippet: `from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from sentence_transformers import CrossEncoder
import asyncio

app = FastAPI()
reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

async def rag_stream(query: str, collection, llm):
    # Step 1: Retrieve (bi-encoder — fast, approximate)
    results = collection.query(query_texts=[query], n_results=10)
    candidates = results['documents'][0]
    
    # Step 2: Rerank (cross-encoder — slow, accurate)
    pairs = [(query, doc) for doc in candidates]
    scores = reranker.predict(pairs)
    top_docs = [candidates[i] for i in scores.argsort()[-3:][::-1]]
    
    # Step 3: Generate with streaming
    context = "\n\n".join(top_docs)
    prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
    
    async for chunk in llm.astream(prompt):
        yield chunk.content  # stream token by token

@app.get("/rag")
async def rag_endpoint(q: str):
    return StreamingResponse(
        rag_stream(q, collection, llm),
        media_type="text/plain"
    )`
      },
      {
        week: 4,
        goal: "Evaluate your RAG system rigorously — this is what separates engineers from hackers",
        topics: "RAG Evaluation + RAGAS Framework",
        subtopics: ["RAGAS metrics: faithfulness, answer relevancy, context precision, context recall", "LLM-as-judge: GPT-4o or Claude grades responses", "Groundedness vs relevance distinction", "Building a golden test set for retrieval", "Iterating on chunk size using eval metrics — not vibes"],
        math: "Faithfulness score = |verified statements| / |total statements in answer|. For 5 test queries, manually compute faithfulness by checking each claim against retrieved docs. This builds intuition for what RAGAS automates.",
        coding: "Run RAGAS evaluation on your RAG system. Create test set of 20 question-answer pairs. Compute all 4 core RAGAS metrics. Print results table.",
        project: "Ablation study: compare RAG performance across chunk_size=[256, 512, 1024] and top_k=[3, 5, 10]. Pick best config. Document in README with data.",
        research: "Read RAGAS paper (arxiv 2309.15217) abstract + metrics section. This is a recent paper you should know and cite in interviews.",
        portfolio: "Month 2 capstone: full RAG system + RAGAS evaluation report. Push everything. Create a LinkedIn post: 'I built a RAG system and evaluated it with RAGAS — here's what I found' + results screenshot.",
        output: "RAGAS evaluation complete. Ablation study documented. LinkedIn post published. Month 2 done.",
        codeSnippet: `from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy, 
    context_precision,
    context_recall,
)
from datasets import Dataset

# Build your test set: 20 QA pairs with ground truth
test_data = {
    "question": ["What is LoRA?", ...],
    "answer": [your_rag_answers],           # your RAG's output
    "contexts": [your_retrieved_contexts],  # what RAG retrieved
    "ground_truth": ["LoRA is a..."]        # correct answers
}

dataset = Dataset.from_dict(test_data)
result = evaluate(
    dataset,
    metrics=[faithfulness, answer_relevancy, 
             context_precision, context_recall]
)

# Target scores to aim for:
# faithfulness > 0.85  (answers grounded in context)
# answer_relevancy > 0.80
# context_precision > 0.75
print(result.to_pandas())`
      }
    ]
  },
  {
    month: 3,
    title: "Agentic AI + LangGraph",
    theme: "Move from pipelines to autonomous systems",
    color: "#FF6B35",
    accent: "#330F00",
    hours: "~84 hrs total · 3 hrs/day",
    bookNote: "Ch.5: 'Focus on an application area… you can do unique work that no one else has done yet.' Agentic AI is that area for 2025-26.",
    weeks: [
      {
        week: 1,
        goal: "Understand tool use and function calling — the atomic unit of agents",
        topics: "LLM Tool Use + Function Calling",
        subtopics: ["OpenAI / Anthropic function calling schemas", "Structured output: JSON mode, Instructor library", "Tool routing: how LLMs decide which tool to call", "Error handling in tool execution", "Parallel tool calling"],
        math: "No new heavy math this week. Instead: review probability — conditional probability P(A|B) and Bayes' theorem. Agents make conditional decisions; understanding this prepares you for MCTS-based agent search.",
        coding: "Build a 3-tool agent using OpenAI API (or local Ollama): web_search tool, calculator tool, code_executor tool. LLM decides which tool to call based on user query.",
        project: "Bengali news summarizer agent: given a topic, agent searches web → fetches Bengali news → summarizes in English. Real use case, portfolio-ready.",
        research: "Read Anthropic's 'Tool use' documentation. Read 'ReAct: Synergizing Reasoning and Acting in Language Models' abstract (arxiv 2210.03629).",
        portfolio: "Push Bengali news agent to GitHub. Record demo GIF using peek or ScreenToGif. Embed in README.",
        output: "Working multi-tool agent. Bengali news demo. ReAct pattern understood.",
        codeSnippet: `from openai import OpenAI  # or use litellm for any backend
import json

client = OpenAI()

tools = [
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "Search the web for current information",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "search query"},
                    "max_results": {"type": "integer", "default": 3}
                },
                "required": ["query"]
            }
        }
    }
]

def run_agent(user_message: str):
    messages = [{"role": "user", "content": user_message}]
    
    while True:  # ReAct loop
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=tools,
            tool_choice="auto"
        )
        
        msg = response.choices[0].message
        if msg.tool_calls:  # LLM wants to use a tool
            for tc in msg.tool_calls:
                result = execute_tool(tc.function.name, 
                                      json.loads(tc.function.arguments))
                messages.append({"role": "tool", 
                                  "tool_call_id": tc.id, 
                                  "content": str(result)})
        else:
            return msg.content  # final answer`
      },
      {
        week: 2,
        goal: "Build controllable, debuggable agents with LangGraph",
        topics: "LangGraph State Machines",
        subtopics: ["Why LangGraph > plain LangChain agents: explicit control flow", "StateGraph: nodes, edges, conditional edges", "Checkpointing: save/resume agent state", "Human-in-the-loop: interrupt and approve before tool use", "Streaming agent steps to frontend"],
        math: "Graph theory basics: directed acyclic graphs (DAGs) vs cyclic graphs. Agent graphs are cyclic (can loop). Understand: why LangGraph uses a state machine model vs a simple chain.",
        coding: "Reimplement your Bengali news agent using LangGraph. Add: human approval step before publishing summary. Add: retry logic if tool fails. Draw the state graph.",
        project: "Research assistant agent: given a research question, agent → searches arXiv → reads abstracts → identifies top 3 papers → generates literature review paragraph.",
        research: "LangGraph documentation: 'Tutorials → Customer Support Bot'. Focus on how interrupt() and checkpoints work. This is used in production at major companies.",
        portfolio: "Add research assistant agent to portfolio. Create architecture diagram showing the state graph (nodes + edges). This demonstrates systems thinking.",
        output: "LangGraph agent with human-in-the-loop. State graph diagram in README. Research assistant working.",
        codeSnippet: `from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from typing import TypedDict, Annotated
import operator

class ResearchState(TypedDict):
    question: str
    search_results: list[str]
    papers: list[dict]
    literature_review: str
    approved: bool  # human-in-the-loop flag

def search_arxiv(state: ResearchState) -> dict:
    results = arxiv_search(state["question"], max_results=10)
    return {"search_results": results}

def rank_papers(state: ResearchState) -> dict:
    # Use LLM to rank by relevance to question
    top3 = llm_rank(state["search_results"], state["question"])
    return {"papers": top3}

def write_review(state: ResearchState) -> dict:
    review = llm_write_review(state["papers"], state["question"])
    return {"literature_review": review}

# Build graph
graph = StateGraph(ResearchState)
graph.add_node("search", search_arxiv)
graph.add_node("rank", rank_papers)
graph.add_node("write", write_review)

graph.set_entry_point("search")
graph.add_edge("search", "rank")
graph.add_edge("rank", "write")
graph.add_edge("write", END)

checkpointer = MemorySaver()
app = graph.compile(
    checkpointer=checkpointer,
    interrupt_before=["write"]  # pause for human approval
)`
      },
      {
        week: 3,
        goal: "Build multi-agent systems — the frontier of production AI",
        topics: "Multi-Agent Architectures",
        subtopics: ["Supervisor agent pattern", "Hierarchical agents: orchestrator + specialists", "Agent communication protocols", "Shared memory vs isolated agents", "CrewAI roles and task delegation", "When NOT to use multi-agent (simpler is better)"],
        math: "Game theory basics: understand Nash equilibrium conceptually. Multi-agent systems where agents compete or cooperate. Read the first 2 pages of 'Multi-Agent Systems' Wikipedia article.",
        coding: "Build a 3-agent CrewAI system: Researcher agent (arXiv search) + Writer agent (summarizer) + Critic agent (fact-checker). Orchestrator coordinates them.",
        project: "Automated conference paper analyzer: upload PDF → Researcher agent reads paper → Writer agent extracts contributions → Critic agent checks claims → outputs structured review JSON.",
        research: "Read 'AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation' abstract (arxiv 2308.08155). Note: where is multi-agent better than single agent?",
        portfolio: "Paper analyzer demo is strong portfolio material — it shows you understand research workflows. Add to GitHub with sample output on your own conference paper.",
        output: "Working multi-agent system. Paper analyzer that processes your own conference paper. CrewAI fluency.",
        codeSnippet: `from crewai import Agent, Task, Crew, Process
from crewai_tools import PDFSearchTool

pdf_tool = PDFSearchTool()

researcher = Agent(
    role="Research Analyst",
    goal="Extract key contributions and methodology from academic papers",
    backstory="Expert at reading CS papers and identifying novel contributions",
    tools=[pdf_tool],
    verbose=True,
    llm="gpt-4o-mini"
)

critic = Agent(
    role="Technical Critic",
    goal="Verify claims made about the paper are factually supported",
    backstory="Skeptical reviewer who checks every claim against paper text",
    verbose=True,
    llm="gpt-4o-mini"
)

analyze_task = Task(
    description="Analyze {paper_path}. Extract: problem, method, results, limitations.",
    expected_output="Structured JSON with keys: problem, method, results, limitations",
    agent=researcher
)

crew = Crew(
    agents=[researcher, critic],
    tasks=[analyze_task],
    process=Process.sequential,
    verbose=True
)`
      },
      {
        week: 4,
        goal: "Build your Month 3 capstone — a complete agentic application",
        topics: "Agentic Application Architecture + Deployment",
        subtopics: ["Production considerations: rate limits, fallbacks, cost tracking", "Streaming agent responses to a Streamlit UI", "Persisting agent memory to a database (SQLite → PostgreSQL)", "Monitoring agent behavior in production", "Cost optimization: when to use GPT-4o-mini vs GPT-4o vs local Ollama"],
        math: "Cost analysis: GPT-4o = $5/1M input tokens, GPT-4o-mini = $0.15/1M. For your agent (avg 3000 tokens/run, 1000 runs/month): compute monthly costs for both. Decision: use mini for retrieval, 4o for final generation.",
        coding: "Build a full-stack agentic app: FastAPI backend + Streamlit frontend + LangGraph agent + ChromaDB memory + wandb tracing. This is your biggest project yet.",
        project: "Bangladesh AI Job Scout: agent scrapes job postings daily → matches to user's skill profile → sends personalized daily email digest. Real value, demonstrates all skills.",
        research: "Read LangSmith documentation — tracing and observability for LangChain/LangGraph apps. Set up tracing on your agent. This is the 'wandb for agents'.",
        portfolio: "Month 3 capstone deployed and documented. Write a LinkedIn post: 'I built an agentic AI system in Bangladesh — here's the architecture'. Tag it with #AIEngineering #LLM.",
        output: "Full-stack agentic application deployed. LangSmith tracing active. LinkedIn post published. You can now call yourself an AI Engineer with 3 shipped projects.",
        codeSnippet: `# Full-stack architecture sketch
# FastAPI backend:
from fastapi import FastAPI, BackgroundTasks
from langgraph.graph import StateGraph

app = FastAPI()

@app.post("/agent/run")
async def run_agent(request: AgentRequest, background: BackgroundTasks):
    thread_id = str(uuid4())
    background.add_task(execute_agent_graph, thread_id, request.query)
    return {"thread_id": thread_id, "status": "running"}

@app.get("/agent/stream/{thread_id}")
async def stream_agent(thread_id: str):
    # Server-Sent Events for real-time streaming
    async def generate():
        async for event in agent_graph.astream_events(
            thread_id=thread_id
        ):
            yield f"data: {event}\n\n"
    return StreamingResponse(generate(), 
                              media_type="text/event-stream")

# Streamlit frontend consumes the SSE stream
# ChromaDB stores conversation memory
# wandb / LangSmith traces every agent step`
      }
    ]
  },
  {
    month: 4,
    title: "MLOps + Production Engineering",
    theme: "Your software background becomes your superpower",
    color: "#B388FF",
    accent: "#1A0033",
    hours: "~84 hrs total · 3 hrs/day",
    bookNote: "Ch.2: 'Your job opportunities will increase if you can also write good software to implement complex AI systems.' MLOps is that software.",
    weeks: [
      {
        week: 1,
        goal: "Instrument every experiment — no more training runs without logs",
        topics: "Experiment Tracking: wandb + MLflow",
        subtopics: ["wandb: runs, sweeps (hyperparameter search), artifacts, reports", "MLflow: experiment tracking, model registry, serving", "Hyperparameter sweeps: grid vs random vs Bayesian optimization", "Comparing runs: wandb parallel coordinates plot", "Model versioning: what changed between v1 and v2?"],
        math: "Bayesian optimization for hyperparameter search: model the objective function as a Gaussian Process. Acquisition function (EI, UCB) decides next point to try. Understand conceptually why it's better than random search.",
        coding: "Retrofit your Month 1 fine-tuning project with wandb sweeps. Tune: learning_rate ∈ [1e-5, 5e-4], lora_r ∈ [8, 16, 32], num_epochs ∈ [2, 5]. Run 15 combinations. Find best config.",
        project: "Create a wandb Report documenting your Bengali LLM experiments: best config, training curves, eval metrics, sample outputs. Share publicly. This is a portfolio artifact.",
        research: "Read wandb's 'A Guide to Hyperparameter Sweeps' documentation. Skim the Optuna paper abstract (arxiv 1907.10902) — the library behind wandb sweeps.",
        portfolio: "Public wandb Report is a strong portfolio signal. Add link to your GitHub README and LinkedIn profile. Very few Bangladesh candidates have this.",
        output: "wandb sweeps complete on Bengali LLM. Best hyperparameters documented. Public wandb Report live.",
        codeSnippet: `import wandb

sweep_config = {
    "method": "bayes",  # Bayesian > random for expensive runs
    "metric": {"name": "eval/rouge_l", "goal": "maximize"},
    "parameters": {
        "learning_rate": {"distribution": "log_uniform_values",
                          "min": 1e-5, "max": 5e-4},
        "lora_r": {"values": [8, 16, 32]},
        "lora_alpha": {"values": [16, 32, 64]},
        "num_train_epochs": {"values": [2, 3, 5]},
        "per_device_train_batch_size": {"values": [4, 8, 16]}
    }
}

def train(config=None):
    with wandb.init(config=config):
        cfg = wandb.config
        model = load_model_with_lora(r=cfg.lora_r, alpha=cfg.lora_alpha)
        trainer = get_trainer(model, lr=cfg.learning_rate, 
                               epochs=cfg.num_train_epochs)
        trainer.train()
        eval_results = trainer.evaluate()
        wandb.log(eval_results)

sweep_id = wandb.sweep(sweep_config, project="bengali-llm-sweeps")
wandb.agent(sweep_id, train, count=15)`
      },
      {
        week: 2,
        goal: "Containerize your models — make them deployable anywhere",
        topics: "Docker + FastAPI Model Serving",
        subtopics: ["Multi-stage Docker builds for ML (minimize image size)", "Model loading optimization: lazy loading, warm-up requests", "FastAPI async endpoints for concurrent inference", "Health checks + readiness probes", "Environment variable management for API keys"],
        math: "Queuing theory basics: Little's Law: L = λW (avg requests in system = arrival rate × avg time). Why this matters: if your model takes 2s/request and you get 10 req/s, you need 20 concurrent capacity minimum.",
        coding: "Dockerize your RAG API. Multi-stage build: builder stage (install deps) + runtime stage (copy only necessary files). Target: image < 3GB. Push to Docker Hub.",
        project: "Deploy containerized RAG API to a free tier: Railway.app, Render.com, or HuggingFace Spaces (Docker). Get a live HTTPS endpoint.",
        research: "Read 'FastAPI in production: best practices' (tiangolo's blog). Focus on: async vs sync endpoints, when to use BackgroundTasks vs Celery.",
        portfolio: "Live deployed API URL is your strongest portfolio asset. Add 'Live Demo' badge to GitHub README with the URL. Recruiters click these.",
        output: "Containerized RAG API deployed to cloud. Docker Hub image public. Live URL in GitHub README.",
        codeSnippet: `# Dockerfile - multi-stage for minimal size
FROM python:3.11-slim as builder
WORKDIR /build
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

FROM python:3.11-slim as runtime
WORKDIR /app
COPY --from=builder /root/.local /root/.local
COPY . .

# Pre-download model at build time (not runtime)
RUN python -c "
from sentence_transformers import SentenceTransformer
SentenceTransformer('BAAI/bge-small-en-v1.5')
print('Model cached')
"

ENV PATH=/root/.local/bin:$PATH
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", 
     "--port", "8000", "--workers", "1"]

# app/main.py - async inference
from contextlib import asynccontextmanager

ml_models = {}  # global model cache

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load models once at startup, not per-request
    ml_models["encoder"] = SentenceTransformer('BAAI/bge-small-en-v1.5')
    ml_models["reranker"] = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    yield
    ml_models.clear()  # cleanup`
      },
      {
        week: 3,
        goal: "Automate quality checks with CI/CD — professional ML engineering standard",
        topics: "CI/CD for ML with GitHub Actions",
        subtopics: ["GitHub Actions: workflow YAML, triggers, jobs, steps", "Automated testing: pytest for ML code (input/output shape tests)", "Model evaluation on PR: block merge if ROUGE-L drops > 5%", "DVC for data versioning", "Automated Docker build + push on tag"],
        math: "Statistical testing: paired t-test to determine if new model is significantly better. If eval metric improves by 2% but variance is high, the improvement may not be real. Understand p-value < 0.05 threshold.",
        coding: "Write GitHub Actions workflow: on PR → run eval on test set → comment results on PR → block merge if metric regresses. This is MLOps engineering, not just scripts.",
        project: "Add CI/CD to your Bengali LLM repo. Every push triggers: 1) unit tests, 2) model eval on 50 test samples, 3) Docker build. Status badges in README.",
        research: "Read DVC documentation (dvc.org) — specifically 'Getting Started: Data Management'. Understanding data versioning is a differentiator for ML Engineer interviews.",
        portfolio: "GitHub Actions CI badge in README = immediate credibility signal. 'Tests passing' means you take quality seriously. Very rare in Bangladesh AI portfolios.",
        output: "CI/CD pipeline active on main ML repo. Automated eval on PR. DVC data versioning initialized.",
        codeSnippet: `# .github/workflows/ml_eval.yml
name: Model Evaluation CI

on:
  pull_request:
    branches: [main]
    paths: ['src/**', 'configs/**']

jobs:
  evaluate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
          cache: 'pip'
      
      - name: Install dependencies
        run: pip install -r requirements.txt
      
      - name: Run evaluation
        env:
          WANDB_API_KEY: \${{ secrets.WANDB_API_KEY }}
          HF_TOKEN: \${{ secrets.HF_TOKEN }}
        run: |
          python scripts/evaluate.py \
            --model_path models/latest \
            --test_data data/test_50.json \
            --output results.json
      
      - name: Check regression
        run: |
          python scripts/check_regression.py \
            --results results.json \
            --threshold 0.75  # block if rougeL < 0.75
      
      - name: Comment results on PR
        uses: actions/github-script@v6
        with:
          script: |
            const results = require('./results.json');
            github.rest.issues.createComment({
              issue_number: context.issue.number,
              body: \`## Eval Results\\nROUGE-L: \${results.rouge_l}\`
            })`
      },
      {
        week: 4,
        goal: "Solidify math foundations — fix the ceiling that blocks senior roles",
        topics: "Linear Algebra + Probability Deep Dive",
        subtopics: ["SVD: W = UΣVᵀ — connection to LoRA (low-rank approximation)", "Eigendecomposition and PCA", "KL divergence: KL(P||Q) = Σ P(x) log(P(x)/Q(x))", "Gaussian distributions and their role in Bayesian ML", "Information theory: entropy H(X) = -Σ p(x) log p(x)"],
        math: "HEAVY WEEK. Do: 1) Compute SVD of a 3×3 matrix by hand. 2) Prove PCA = SVD on centered data. 3) Compute KL divergence between two Gaussians analytically. 4) Watch 3Blue1Brown LA series episodes 13-15.",
        coding: "Implement PCA from scratch using SVD (numpy.linalg.svd). Apply to MNIST — reduce 784 dims to 50. Visualize explained variance ratio. Compare sklearn PCA — should match.",
        project: "No new project this week. Refactor/document Month 1-3 projects. Write clean READMEs. Add math intuition sections explaining why you made key design choices.",
        research: "Work through MIT 18.06 Linear Algebra notes — lectures 14-16 (eigenvalues, SVD). Free at ocw.mit.edu. These appear directly in ML research papers.",
        portfolio: "Month 4 doesn't add a project — it adds depth to existing ones. Update your GitHub profile README with a skills section. Add: 'Strong in: transformers, RAG, agentic AI, MLOps'.",
        output: "Math gaps closed on LA and probability. PCA from scratch implemented. 4 projects cleaned and re-documented. GitHub profile polished.",
        codeSnippet: `import numpy as np
from sklearn.datasets import fetch_openml

# PCA from scratch via SVD
# Theorem: PCA directions = right singular vectors of centered data matrix X
def pca_from_scratch(X: np.ndarray, n_components: int):
    # Center the data: subtract mean
    X_centered = X - X.mean(axis=0)  # (n_samples, n_features)
    
    # SVD: X = UΣVᵀ
    # U: (n,n) left singular vectors
    # S: singular values (sqrt of eigenvalues of XᵀX)  
    # Vt: (p,p) right singular vectors = PCA directions
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
    
    # Principal components = first n_components rows of Vt
    components = Vt[:n_components]  # (n_components, n_features)
    
    # Project data onto components
    X_reduced = X_centered @ components.T  # (n_samples, n_components)
    
    # Explained variance ratio
    explained_variance = (S ** 2) / (len(X) - 1)
    ev_ratio = explained_variance / explained_variance.sum()
    
    return X_reduced, components, ev_ratio[:n_components]

# KL divergence between two Gaussians:
# KL(N(μ₁,σ₁²) || N(μ₂,σ₂²)) = log(σ₂/σ₁) + (σ₁²+(μ₁-μ₂)²)/(2σ₂²) - 1/2
def kl_gaussian(mu1, sigma1, mu2, sigma2):
    return (np.log(sigma2/sigma1) + 
            (sigma1**2 + (mu1-mu2)**2) / (2*sigma2**2) - 0.5)`
      }
    ]
  },
  {
    month: 5,
    title: "Specialization + Research Depth",
    theme: "Go deep in one area — breadth got you here, depth gets you hired",
    color: "#FFD700",
    accent: "#332800",
    hours: "~84 hrs total · 3 hrs/day",
    bookNote: "Ch.2: 'I've known many ML engineers who benefited from deeper skills in an application area such as NLP or computer vision.' Bengali NLP is yours.",
    weeks: [
      {
        week: 1,
        goal: "Understand RLHF and alignment — every LLM company needs this",
        topics: "RLHF + DPO + Alignment",
        subtopics: ["RLHF pipeline: SFT → reward model → PPO", "DPO: Direct Preference Optimization (simpler than RLHF)", "Constitutional AI (Anthropic's approach)", "Reward hacking and Goodhart's Law in ML context", "TRL library: SFTTrainer, RewardTrainer, DPOTrainer"],
        math: "DPO loss: L_DPO = -E[log σ(β log(π_θ(y_w|x)/π_ref(y_w|x)) - β log(π_θ(y_l|x)/π_ref(y_l|x)))]. Trace through: y_w = preferred, y_l = rejected. β controls KL penalty strength. Understand why this works without explicit reward model.",
        coding: "Run DPO fine-tuning on your Bengali model using TRL's DPOTrainer. Create 100 preference pairs: (prompt, chosen_response, rejected_response). Track reward margin in wandb.",
        project: "Bengali instruction-following dataset creation: write 200 Bengali prompts, generate 2 responses each from your fine-tuned model, label preferred. This dataset is itself a research contribution.",
        research: "Read DPO paper (arxiv 2305.18290) sections 1-3. Read 'Training language models to follow instructions with human feedback' (InstructGPT, arxiv 2203.02155) abstract + section 3.",
        portfolio: "Publish your Bengali preference dataset to HuggingFace Hub. First Bengali DPO dataset from Bangladesh = genuine research contribution. Add to CV under 'Datasets'.",
        output: "DPO-trained Bengali model. Preference dataset on HuggingFace Hub. RLHF/DPO concepts interview-ready.",
        codeSnippet: `from trl import DPOTrainer, DPOConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("your-finetuned-bengali-model")
ref_model = AutoModelForCausalLM.from_pretrained("your-finetuned-bengali-model")
tokenizer = AutoTokenizer.from_pretrained("your-finetuned-bengali-model")

# Dataset format for DPO
# {"prompt": "...", "chosen": "preferred response", "rejected": "bad response"}

training_args = DPOConfig(
    beta=0.1,           # KL penalty — higher = stay closer to reference
    learning_rate=5e-7, # much lower than SFT — small adjustments
    num_train_epochs=1,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    output_dir="./dpo_output",
    report_to="wandb",
)

dpo_trainer = DPOTrainer(
    model=model,
    ref_model=ref_model,  # frozen reference (original SFT model)
    args=training_args,
    train_dataset=preference_dataset,
    tokenizer=tokenizer,
)
dpo_trainer.train()
# Monitor: reward_margin (chosen_reward - rejected_reward) should increase`
      },
      {
        week: 2,
        goal: "Bengali NLP specialization — your unique competitive advantage",
        topics: "Low-Resource NLP + Bengali Language AI",
        subtopics: ["Transfer learning for low-resource languages", "Bengali tokenization challenges: morphologically rich", "BanglaBERT, Bangla-BERT models on HuggingFace", "Cross-lingual models: mBERT, XLM-RoBERTa for Bengali", "Indic NLP Library for Bengali preprocessing"],
        math: "Cross-lingual transfer: understand why multilingual models share embedding space. Read about 'curse of multilinguality' — adding more languages dilutes per-language capacity. Compute: for 100 languages vs 1 language, how much capacity per language?",
        coding: "Fine-tune XLM-RoBERTa on Bengali NER (Named Entity Recognition) using BanglaKit dataset. Evaluate with seqeval. Push to HuggingFace Hub.",
        project: "Bengali Fake News Detector: fine-tune BanglaBERT on Bengali news dataset. Build Streamlit UI. This is socially impactful and technically strong.",
        research: "Search Google Scholar: 'Bengali NLP 2024 2025'. Find 3 recent papers. Write a 1-page literature review. This is prep for your second conference paper.",
        portfolio: "Bengali Fake News Detector is your most shareable project: relevant, Bengali audience can use it, shows domain expertise. Create a proper demo with Gradio.",
        output: "Bengali NER model on HuggingFace Hub. Fake News Detector live demo. 1-page literature review written.",
        codeSnippet: `from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import DataCollatorForTokenClassification
import evaluate

model_name = "csebuetnlp/banglabert"  # BanglaBERT
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(
    model_name, 
    num_labels=len(label_list)  # BIO tags: B-PER, I-PER, B-ORG, etc.
)

# Bengali NER labels
label_list = ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 
              'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC']

# seqeval for proper NER evaluation (entity-level, not token-level)
seqeval = evaluate.load("seqeval")

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)
    
    true_predictions = [
        [label_list[p] for p, l in zip(pred, lab) if l != -100]
        for pred, lab in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for p, l in zip(pred, lab) if l != -100]
        for pred, lab in zip(predictions, labels)
    ]
    results = seqeval.compute(predictions=true_predictions, 
                               references=true_labels)
    return {"f1": results["overall_f1"], 
            "precision": results["overall_precision"],
            "recall": results["overall_recall"]}`
      },
      {
        week: 3,
        goal: "Implement a research paper — the most respected portfolio signal",
        topics: "Paper Implementation",
        subtopics: ["How to read a paper efficiently: abstract → conclusion → figures → method", "Reproducing results: why published numbers are hard to match", "Ablation studies: what components actually matter", "Writing a technical blog post about your implementation", "Connecting implementation to your Bengali NLP work"],
        math: "Pick a paper with math you can now handle: FlashAttention (memory-efficient attention), Mixture of Experts (MoE routing), or Speculative Decoding. Work through the key equations section.",
        coding: "Implement FlashAttention-style tiling in pure PyTorch (not CUDA). It won't be as fast as the real thing but proves you understand the algorithm. Or: implement MoE routing from scratch.",
        project: "Write a technical blog post (1500 words) on Medium/dev.to: 'I implemented [paper] from scratch — here's what I learned'. Include: your code, results vs paper, key insights.",
        research: "Read 'FlashAttention: Fast and Memory-Efficient Exact Attention' (arxiv 2205.14135) sections 1-3. Or: 'Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer'.",
        portfolio: "Published technical blog post + GitHub implementation repo = two portfolio items from one week. This is the highest signal-per-hour activity for research roles.",
        output: "Paper implementation on GitHub. Blog post published. Month 5 research depth visible to any recruiter who checks your profile.",
        codeSnippet: `import torch
import torch.nn.functional as F
import math

# FlashAttention key insight: tile Q, K, V to avoid materializing full N×N matrix
# Standard attention: O(N²) memory. FlashAttention: O(N) memory.
# This is why LLMs can handle 128K context windows.

def flash_attention_naive_tiled(Q, K, V, tile_size=64):
    """
    Educational implementation of FlashAttention tiling concept.
    NOT optimized — shows the algorithm, not the CUDA kernels.
    
    Q, K, V: (batch, heads, seq_len, d_k)
    """
    B, H, N, d = Q.shape
    scale = 1.0 / math.sqrt(d)
    
    output = torch.zeros_like(Q)
    
    # Tile over sequence dimension
    for i in range(0, N, tile_size):
        Qi = Q[:, :, i:i+tile_size, :]  # (B, H, tile, d)
        
        # Accumulate over K,V tiles
        mi = torch.full((B, H, Qi.shape[2], 1), -torch.inf)  # running max
        li = torch.zeros(B, H, Qi.shape[2], 1)  # running sum
        Oi = torch.zeros_like(Qi)
        
        for j in range(0, N, tile_size):
            Kj = K[:, :, j:j+tile_size, :]
            Vj = V[:, :, j:j+tile_size, :]
            
            Sij = torch.matmul(Qi, Kj.transpose(-2, -1)) * scale
            mij = Sij.max(dim=-1, keepdim=True).values
            Pij = torch.exp(Sij - mij)  # numerically stable softmax
            lij = Pij.sum(dim=-1, keepdim=True)
            
            # Update running statistics (online softmax)
            mi_new = torch.maximum(mi, mij)
            li_new = torch.exp(mi - mi_new) * li + torch.exp(mij - mi_new) * lij
            Oi = (torch.exp(mi - mi_new) * li * Oi + 
                  torch.exp(mij - mi_new) * torch.matmul(Pij, Vj)) / li_new
            mi, li = mi_new, li_new
        
        output[:, :, i:i+tile_size, :] = Oi
    
    return output`
      },
      {
        week: 4,
        goal: "Second conference paper — extend your first paper with AI/ML contribution",
        topics: "Research Writing + Conference Submission Strategy",
        subtopics: ["How to extend your existing conference paper", "ACL, EMNLP, COLING — NLP conference deadlines and tiers", "ICCIT Bangladesh — local conference, high acceptance, good for CV", "Paper structure: intro → related work → method → experiments → conclusion", "Using your Bengali NLP work as a research contribution"],
        math: "Statistical significance testing for your paper: compute paired bootstrap test or McNemar's test to prove your model is significantly better than baseline. This is required for NLP papers.",
        coding: "Write experiment scripts for your second paper: baseline comparison, ablation study (each component's contribution), error analysis. Everything reproducible from a single script.",
        project: "Draft 4-page paper: 'Bengali [Task] using [Your Method]: A LoRA-based Approach'. Target: ICCIT 2025 (Bangladesh) or ACL workshops (international).",
        research: "Read 3 papers from ACL Anthology on Bengali NLP. Study their paper structure. Your paper should follow the same structure. Note their 'Related Work' section for citations.",
        portfolio: "A second conference paper in submission = 'active researcher' on your CV. Even a preprint on arXiv demonstrates research activity. Submit to arXiv immediately after first draft.",
        output: "Paper draft complete. Submitted to arXiv. Conference submission target identified. Month 5 done: you are now an AI Engineer with research depth.",
        codeSnippet: `# Experiment script template for reproducible research
# Every result in your paper must be reproducible from this script

import argparse
import json
import random
import numpy as np
import torch
from pathlib import Path

def set_seed(seed: int):
    """Reproducibility — always do this in research code."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def run_experiment(config: dict) -> dict:
    set_seed(config["seed"])
    
    model = load_model(config["model_name"], config["lora_r"])
    dataset = load_dataset(config["dataset"])
    
    trainer = get_trainer(model, dataset, config)
    trainer.train()
    
    eval_results = trainer.evaluate()
    
    # Save everything for paper tables
    results = {
        "config": config,
        "eval_metrics": eval_results,
        "model_path": config["output_dir"],
        "timestamp": datetime.now().isoformat()
    }
    
    Path(config["results_dir"]).mkdir(exist_ok=True)
    with open(f"{config['results_dir']}/results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    return results

if __name__ == "__main__":
    # Table 1: Baseline comparison
    # Table 2: Ablation study (vary lora_r)
    # Table 3: Low-resource (vary training set size)
    for exp_config in load_experiment_configs("configs/"):
        results = run_experiment(exp_config)
        print(f"{exp_config['name']}: F1={results['eval_metrics']['eval_f1']:.4f}")`
      }
    ]
  },
  {
    month: 6,
    title: "Job Search + Applications",
    theme: "Convert 5 months of work into opportunities",
    color: "#FF4F8B",
    accent: "#330015",
    hours: "~84 hrs total · 3 hrs/day",
    bookNote: "Ch.7: 'Switching either roles or industries [is] easier than doing both at once.' You're doing a role switch only — you stay in tech. Easiest transition path.",
    weeks: [
      {
        week: 1,
        goal: "Build a recruiter-ready profile in 1 week",
        topics: "Resume + LinkedIn + GitHub Positioning",
        subtopics: ["Resume: AI/ML engineer narrative, not web dev", "Quantify everything: 'fine-tuned LLM achieving 23% ROUGE-L improvement'", "GitHub profile README as portfolio homepage", "LinkedIn: headline, about section, featured projects", "Cold outreach message template for informational interviews"],
        math: "No math this week. Pure execution.",
        coding: "Build a personal portfolio site with Streamlit or Next.js (1 day max). Must include: project demos, GitHub links, contact form. Deploy free on Streamlit Cloud.",
        project: "Create 5-minute Loom video walkthrough of your best project (RAG system or agentic app). This goes in LinkedIn 'Featured' section and replaces 1000 words of explanation.",
        research: "Research 20 target companies: 10 Bangladesh tech companies with AI teams (Brain Station 23, BJIT, Kaz Software, DataSoft, Therap, Shajgoj), 10 remote-first AI startups (Hugging Face, Replicate, Together AI, Mistral ecosystem companies).",
        portfolio: "By end of week: resume v1, LinkedIn updated, GitHub polished, target company list compiled, portfolio site live.",
        output: "Recruiter-ready profile across all platforms. Target list of 20 companies. Portfolio site deployed.",
        codeSnippet: `# Resume bullet formula: [Action verb] + [what] + [how] + [metric]
# BAD:  "Worked on NLP projects using transformers"
# GOOD: "Fine-tuned LLaMA-3.2 with QLoRA on Bengali instruction dataset, 
#         achieving 31% ROUGE-L improvement over base model (eval on 500 samples)"

# BAD:  "Built a RAG system"
# GOOD: "Engineered production RAG pipeline (ChromaDB + cross-encoder reranking + 
#         FastAPI streaming) achieving 0.87 RAGAS faithfulness score on domain QA"

# BAD:  "Experience with ML deployment"
# GOOD: "Containerized and deployed 3 ML APIs (Docker + Railway) with CI/CD 
#         pipeline: automated eval gates block regressions on every PR"

# GitHub README structure (copy this):
# ## Projects
# | Project | Tech Stack | Demo | Paper |
# |---------|-----------|------|-------|
# | Bengali LLM | LoRA, QLoRA, TRL | [HuggingFace] | [arXiv] |
# | RAG System  | ChromaDB, FastAPI | [Live API] | - |
# | Agent App   | LangGraph, CrewAI | [Demo] | - |`
      },
      {
        week: 2,
        goal: "Informational interviews — intelligence gathering before applying",
        topics: "Network + Informational Interviews",
        subtopics: ["Cold outreach on LinkedIn: 3-sentence message format", "Questions to ask in informational interview (from Ch.8)", "How to approach Bangladesh company AI teams", "Remote job communities: Remotive, AI jobs board", "Building in public: Twitter/LinkedIn posts attract inbound"],
        math: "No math. Probability mindset only: if 10% of cold outreach converts to calls, and you need 5 calls, send 50 messages. Start now.",
        coding: "Build a job application tracker in Airtable or Notion. Columns: company, role, status, contact, next action, deadline. Treat job search as a pipeline.",
        project: "Write 2 LinkedIn posts this week: one technical (share your paper implementation insight), one story-based (your journey from web dev to AI). These attract inbound recruiter messages.",
        research: "Read Ch.8 of the uploaded book — informational interview questions. Prepare YOUR answers to these questions (because interviewers ask them back). The book says these are underused.",
        portfolio: "Send 20 LinkedIn connection requests + messages to AI engineers at your 20 target companies. Template: 'Hi [name], I noticed you work on [specific thing at company]. I recently [specific thing you built]. Would you spare 20 minutes to share how your team uses [relevant tech]?'",
        output: "5+ informational interviews scheduled. Job tracker setup with 20 companies. 2 LinkedIn posts published. Network expanding.",
        codeSnippet: `# Outreach message template (LinkedIn, < 300 chars):
"""
Hi [Name],

I noticed you're working on [specific project/tech] at [Company].
I recently built a Bengali LLM fine-tuning pipeline + RAG system 
and published a paper on [topic] — I'm transitioning from 
full-stack to AI engineering.

Would you spare 20 minutes to share how your AI team is structured 
and what skills matter most for your stack?

Thanks for your time.
"""

# Questions to ask in informational interview (from Ch.8 of book):
questions = [
    "What does your typical week look like?",
    "What skills do you wish you had when you started?",
    "What does your tech stack look like for LLM products?",
    "How does your team evaluate model quality?",
    "What separates strong ML Engineer candidates from average ones?",
    "Is the team hiring? Would you be open to referring me if I'm a fit?",
]
# Note: last question only if the conversation went well`
      },
      {
        week: 3,
        goal: "Technical interview preparation — systems design + ML fundamentals",
        topics: "Interview Preparation",
        subtopics: ["ML system design: design a Bengali QA system at scale", "Coding: ML-adjacent LeetCode (medium level), focus on trees, graphs, DP", "ML fundamentals: bias-variance, regularization, when to use what model", "LLM-specific questions: explain attention, why LoRA works, how RAG differs from fine-tuning", "Behavioral: STAR format for your projects"],
        math: "Review: gradient descent, Adam optimizer derivation, backpropagation chain rule. These are common interview questions at serious ML companies.",
        coding: "Practice 10 LeetCode mediums: focus on graphs (BFS/DFS) and dynamic programming — these appear in ML algorithm interviews. Time yourself: 25 min max per problem.",
        project: "Prepare 3 'project deep dives': for each of your top 3 projects, prepare: problem → approach → key decision (and why) → results → what you'd do differently. This is most of the interview.",
        research: "Read Chip Huyen's 'ML Interviews Book' (free at huyenchip.com/ml-interviews-book). Focus on Chapter 2: Math, Chapter 5: Computer Science. These match real interview questions.",
        portfolio: "Mock interview: ask a friend (or use Pramp.com) to give you a 45-min ML system design interview. Record it. Watch it back. This is painful but essential.",
        output: "3 project deep-dives prepared. 10 LeetCode mediums complete. 1 mock interview done. Interview-ready.",
        codeSnippet: `# Common LLM interview questions — prepare these answers:

# Q1: "Explain attention mechanism"
# A: Attention(Q,K,V) = softmax(QKᵀ/√dₖ)V
#    Q,K,V are linear projections of input. softmax gives 
#    importance weights. √dₖ prevents vanishing gradients 
#    in high dimensions (dot products grow with d).

# Q2: "Why LoRA instead of full fine-tuning?"  
# A: Weight updates ΔW = BA where rank(B,A) = r << min(d,k)
#    Assumes updates lie in low-dimensional subspace.
#    99%+ parameter reduction → fits on consumer GPU.
#    Merge at inference: W' = W₀ + BA → zero latency cost.

# Q3: "RAG vs fine-tuning — when to use which?"
# A: Fine-tune when: task/style change, fixed knowledge domain
#    RAG when: knowledge updates frequently, need citations,
#              multi-document reasoning, limited GPU budget
#    Hybrid: fine-tune for style/format, RAG for knowledge

# Q4: "Design a Bengali QA system at scale"
# A: Ingest → chunk → embed (BGE-M3 multilingual) → ChromaDB
#    Retrieve top-10 → rerank (cross-encoder) → BanglaBERT or
#    Bengali-tuned LLM → generate. 
#    Scale: FAISS IVF index (approximate), async FastAPI,
#    Redis cache for frequent queries, monitoring with LangSmith`
      },
      {
        week: 4,
        goal: "Apply, negotiate, and plan Month 7+",
        topics: "Active Applications + Offer Handling + MS/PhD Track",
        subtopics: ["Tailoring applications: each JD has 5-7 keywords, match them exactly", "Referrals > cold applications (from informational interviews)", "Salary research: Bangladesh AI Engineer market rates", "MS/PhD SOP: how your projects and paper tell a research story", "Offer negotiation: market data > emotions"],
        math: "Expected value calculation for job offers: factor in salary, growth rate, learning opportunity, team quality. Don't optimize only for salary number.",
        coding: "Build a simple AI-powered cover letter generator using your own RAG system: input job description → retrieves your relevant projects → generates tailored cover letter. This is a meta-project that uses your skills to get a job.",
        project: "MS/PhD Statement of Purpose first draft: 500 words. Story: web dev → got interested in AI → built X, Y, Z → published paper → want to research [specific problem] → this program because [specific lab/professor].",
        research: "Research 5 MS/PhD programs: BUET CSE, NSU CSE, Asian Institute of Technology (Thailand), NTU Singapore, any European program with AI focus. Check professor research interests — contact them directly before applying.",
        portfolio: "Month 6 final state: 5 deployed projects, 1 published/submitted paper, 20 applications sent, 5+ informational interviews completed, MS/PhD SOP drafted. This is your 6-month outcome.",
        output: "20+ applications active. SOP drafted. At least 1 interview scheduled. Clear next-90-day plan regardless of immediate outcome.",
        codeSnippet: `# Cover letter generator using your own RAG system
# Meta: use what you built to get the job

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# Your portfolio is stored in ChromaDB from Month 2
def generate_cover_letter(job_description: str, your_profile: dict):
    # Retrieve relevant projects for this JD
    relevant_projects = portfolio_rag.invoke(job_description)
    
    prompt = ChatPromptTemplate.from_template("""
    You are writing a cover letter for {name}.
    
    Job Description:
    {job_description}
    
    Relevant projects and experience:
    {relevant_projects}
    
    Publications: {papers}
    
    Write a concise, specific 3-paragraph cover letter that:
    1. Opens with a specific technical achievement relevant to this role
    2. Connects 2-3 projects directly to job requirements  
    3. Closes with clear next step
    
    Avoid: generic phrases, "I am passionate about", clichés.
    """)
    
    chain = prompt | ChatOpenAI(model="gpt-4o-mini")
    return chain.invoke({
        "name": your_profile["name"],
        "job_description": job_description,
        "relevant_projects": relevant_projects,
        "papers": your_profile["papers"]
    })`
      }
    ]
  }
];

const MonthTab = ({ month, title, isActive, onClick, color }) => (
  <button
    onClick={onClick}
    style={{
      background: isActive ? color : "transparent",
      border: `1px solid ${isActive ? color : "#2a2a3e"}`,
      color: isActive ? "#000" : "#888",
      padding: "8px 14px",
      borderRadius: "6px",
      cursor: "pointer",
      fontSize: "12px",
      fontFamily: "'Space Mono', monospace",
      fontWeight: isActive ? "700" : "400",
      transition: "all 0.2s",
      whiteSpace: "nowrap",
      letterSpacing: "0.03em"
    }}
  >
    M{month}
  </button>
);

const WeekCard = ({ week, color, expanded, onToggle }) => {
  const accent = "#0d0d1a";
  return (
    <div style={{
      border: `1px solid ${expanded ? color : "#1e1e2e"}`,
      borderRadius: "10px",
      marginBottom: "12px",
      overflow: "hidden",
      background: expanded ? "#0a0a1a" : "#070711",
      transition: "all 0.3s"
    }}>
      <button
        onClick={onToggle}
        style={{
          width: "100%",
          background: "transparent",
          border: "none",
          padding: "14px 16px",
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          cursor: "pointer",
          textAlign: "left"
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: "12px" }}>
          <span style={{
            background: color,
            color: "#000",
            borderRadius: "4px",
            padding: "2px 8px",
            fontSize: "10px",
            fontFamily: "'Space Mono', monospace",
            fontWeight: "700",
            letterSpacing: "0.05em"
          }}>W{week.week}</span>
          <span style={{ color: "#e0e0f0", fontSize: "13px", fontWeight: "600" }}>{week.goal}</span>
        </div>
        <span style={{ color: color, fontSize: "16px", flexShrink: 0 }}>{expanded ? "▲" : "▼"}</span>
      </button>

      {expanded && (
        <div style={{ padding: "0 16px 16px", display: "flex", flexDirection: "column", gap: "14px" }}>

          {/* Topics */}
          <Section label="TOPICS" color={color}>
            <p style={{ color: "#c0c0e0", fontSize: "13px", margin: "0 0 6px 0", fontWeight: "600" }}>{week.topics}</p>
            <ul style={{ margin: 0, padding: "0 0 0 16px" }}>
              {week.subtopics.map((s, i) => (
                <li key={i} style={{ color: "#888", fontSize: "12px", marginBottom: "3px" }}>{s}</li>
              ))}
            </ul>
          </Section>

          {/* 2-col grid */}
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "10px" }}>
            <Section label="📐 MATH" color={color} small>
              <p style={{ color: "#b0b0cc", fontSize: "12px", margin: 0, lineHeight: "1.5" }}>{week.math}</p>
            </Section>
            <Section label="💼 JOB / PORTFOLIO" color={color} small>
              <p style={{ color: "#b0b0cc", fontSize: "12px", margin: 0, lineHeight: "1.5" }}>{week.portfolio}</p>
            </Section>
          </div>

          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "10px" }}>
            <Section label="🔬 RESEARCH" color={color} small>
              <p style={{ color: "#b0b0cc", fontSize: "12px", margin: 0, lineHeight: "1.5" }}>{week.research}</p>
            </Section>
            <Section label="🏗️ PROJECT" color={color} small>
              <p style={{ color: "#b0b0cc", fontSize: "12px", margin: 0, lineHeight: "1.5" }}>{week.project}</p>
            </Section>
          </div>

          {/* Coding */}
          <Section label="💻 CODING TASK" color={color}>
            <p style={{ color: "#b0b0cc", fontSize: "12px", margin: 0, lineHeight: "1.5" }}>{week.coding}</p>
          </Section>

          {/* Code snippet */}
          <div style={{
            background: "#050510",
            borderRadius: "8px",
            padding: "12px",
            border: `1px solid #1a1a2e`,
            position: "relative"
          }}>
            <span style={{
              position: "absolute",
              top: "8px",
              right: "10px",
              fontSize: "9px",
              fontFamily: "'Space Mono', monospace",
              color: "#444",
              letterSpacing: "0.1em"
            }}>PYTHON</span>
            <pre style={{
              color: "#7dd3fc",
              fontSize: "10px",
              margin: 0,
              overflowX: "auto",
              fontFamily: "'Space Mono', monospace",
              lineHeight: "1.6",
              whiteSpace: "pre-wrap",
              wordBreak: "break-word"
            }}>{week.codeSnippet}</pre>
          </div>

          {/* Output */}
          <div style={{
            background: `${color}11`,
            border: `1px solid ${color}44`,
            borderRadius: "8px",
            padding: "10px 14px",
            display: "flex",
            alignItems: "flex-start",
            gap: "8px"
          }}>
            <span style={{ fontSize: "14px", flexShrink: 0 }}>✅</span>
            <div>
              <span style={{ color: color, fontSize: "10px", fontFamily: "'Space Mono', monospace", letterSpacing: "0.1em", display: "block", marginBottom: "3px" }}>WEEK OUTPUT</span>
              <p style={{ color: "#d0d0f0", fontSize: "12px", margin: 0, lineHeight: "1.5" }}>{week.output}</p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

const Section = ({ label, color, children, small }) => (
  <div style={{
    background: "#0a0a1a",
    border: "1px solid #1a1a2e",
    borderRadius: "8px",
    padding: small ? "10px" : "12px"
  }}>
    <span style={{
      color: color,
      fontSize: "9px",
      fontFamily: "'Space Mono', monospace",
      letterSpacing: "0.12em",
      display: "block",
      marginBottom: "6px",
      fontWeight: "700"
    }}>{label}</span>
    {children}
  </div>
);

export default function AIRoadmap() {
  const [activeMonth, setActiveMonth] = useState(0);
  const [expandedWeeks, setExpandedWeeks] = useState({ 0: true });

  const month = roadmapData[activeMonth];

  const toggleWeek = (weekIdx) => {
    setExpandedWeeks(prev => ({ ...prev, [weekIdx]: !prev[weekIdx] }));
  };

  return (
    <div style={{
      background: "#030308",
      minHeight: "100vh",
      fontFamily: "'Inter', system-ui, sans-serif",
      color: "#e0e0f0",
      padding: "0"
    }}>
      {/* Header */}
      <div style={{
        background: "#07070f",
        borderBottom: "1px solid #1a1a2e",
        padding: "20px 20px 16px",
        position: "sticky",
        top: 0,
        zIndex: 100
      }}>
        <div style={{ display: "flex", alignItems: "flex-start", justifyContent: "space-between", marginBottom: "14px", flexWrap: "wrap", gap: "8px" }}>
          <div>
            <h1 style={{
              margin: "0 0 4px 0",
              fontSize: "18px",
              fontWeight: "800",
              letterSpacing: "-0.02em",
              color: "#fff"
            }}>AI Career Roadmap</h1>
            <p style={{ margin: 0, fontSize: "11px", color: "#555", fontFamily: "'Space Mono', monospace" }}>
              Full-Stack → AI Engineer · 3 hrs/day · 6 months · Based on Andrew Ng's framework
            </p>
          </div>
          <div style={{
            background: `${month.color}22`,
            border: `1px solid ${month.color}55`,
            borderRadius: "20px",
            padding: "4px 12px",
            fontSize: "10px",
            color: month.color,
            fontFamily: "'Space Mono', monospace"
          }}>
            {month.hours}
          </div>
        </div>

        {/* Month tabs */}
        <div style={{ display: "flex", gap: "6px", overflowX: "auto", paddingBottom: "2px" }}>
          {roadmapData.map((m, i) => (
            <MonthTab
              key={i}
              month={m.month}
              title={m.title}
              isActive={activeMonth === i}
              onClick={() => { setActiveMonth(i); setExpandedWeeks({ 0: true }); }}
              color={m.color}
            />
          ))}
        </div>
      </div>

      {/* Month header */}
      <div style={{
        padding: "16px 20px",
        background: `linear-gradient(135deg, ${month.color}08, transparent)`,
        borderBottom: "1px solid #0f0f1e"
      }}>
        <div style={{ display: "flex", alignItems: "flex-start", gap: "12px", flexWrap: "wrap" }}>
          <div style={{
            background: month.color,
            color: "#000",
            borderRadius: "8px",
            padding: "4px 12px",
            fontSize: "11px",
            fontFamily: "'Space Mono', monospace",
            fontWeight: "800",
            flexShrink: 0
          }}>
            MONTH {month.month}
          </div>
          <div>
            <h2 style={{ margin: "0 0 3px 0", fontSize: "16px", fontWeight: "700", color: "#fff" }}>{month.title}</h2>
            <p style={{ margin: "0 0 6px 0", fontSize: "12px", color: "#666", fontStyle: "italic" }}>{month.theme}</p>
          </div>
        </div>
        <div style={{
          marginTop: "10px",
          background: "#0a0a18",
          borderLeft: `3px solid ${month.color}`,
          padding: "8px 12px",
          borderRadius: "0 6px 6px 0"
        }}>
          <p style={{ margin: 0, fontSize: "11px", color: "#888", fontStyle: "italic", lineHeight: "1.5" }}>
            📖 {month.bookNote}
          </p>
        </div>
      </div>

      {/* Weeks */}
      <div style={{ padding: "16px 20px" }}>
        {month.weeks.map((week, i) => (
          <WeekCard
            key={i}
            week={week}
            color={month.color}
            expanded={!!expandedWeeks[i]}
            onToggle={() => toggleWeek(i)}
          />
        ))}
      </div>

      {/* Month progress indicators */}
      <div style={{
        padding: "0 20px 20px",
        display: "grid",
        gridTemplateColumns: "repeat(6, 1fr)",
        gap: "6px"
      }}>
        {roadmapData.map((m, i) => (
          <div
            key={i}
            onClick={() => { setActiveMonth(i); setExpandedWeeks({ 0: true }); window.scrollTo(0, 0); }}
            style={{
              background: i <= activeMonth ? `${m.color}22` : "#0a0a15",
              border: `1px solid ${i === activeMonth ? m.color : i < activeMonth ? m.color + "44" : "#1a1a2e"}`,
              borderRadius: "8px",
              padding: "8px",
              cursor: "pointer",
              textAlign: "center",
              transition: "all 0.2s"
            }}
          >
            <div style={{ fontSize: "9px", color: i === activeMonth ? m.color : "#444", fontFamily: "'Space Mono', monospace", letterSpacing: "0.05em" }}>M{m.month}</div>
            <div style={{ fontSize: "8px", color: "#444", marginTop: "2px", lineHeight: "1.3" }}>{m.title.split(" ").slice(0, 2).join(" ")}</div>
          </div>
        ))}
      </div>
    </div>
  );
}
