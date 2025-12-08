# Search NEU Agentic Course Registration Assistant
### CS 4100 F25 Final Project

An AI-powered course search tool for Northeastern University undergrad students that uses TF-IDF, semantic embeddings, and ReAct agents to help students find relevant courses.

## Abstract

Students at Northeastern University face challenges finding courses that match their academic interests and requirements due to the large course catalog and complex filtering needs. This project addresses this problem by developing an intelligent course search system that combines traditional keyword search (TF-IDF), semantic search using embeddings, and an AI agent-based approach (ReAct framework) to help students discover relevant courses. The system allows students to search using natural language queries, filter by major requirements, course level, credits, and department, and receive intelligent recommendations. Key outcomes include a functional web application that demonstrates the effectiveness of combining multiple search paradigms and the ReAct agent framework for educational information retrieval.

## Overview

### What is the problem?

Course selection is a critical decision for students, but navigating large university course catalogs can be overwhelming. Students need to find courses that align with their interests, meet degree requirements, fit their schedule, and match their academic level. Traditional course search interfaces often require students to know specific course codes or use exact keyword matches, which limits discovery of relevant courses. Additionally, students may struggle to understand how courses relate to their broader learning goals or major requirements.

### Why is this problem interesting?
This problem is interesting because it addresses a real-world information retrieval challenge that affects thousands of students annually. Effective course discovery can improve student satisfaction, help students make better academic decisions, and potentially increase course enrollment in underutilized but valuable courses. The problem also serves as an excellent testbed for comparing different information retrieval approaches—from traditional keyword-based methods to modern semantic search and AI agent systems. 

### What is the approach you propose to tackle the problem?

We built a multi-paradigm search system that combines three complementary approaches:

1. **TF-IDF Keyword Search**: Traditional term frequency-inverse document frequency search that excels at exact keyword matching and course code searches.

2. **Semantic Search**: Vector-based search using sentence transformers (all-mpnet-base-v2) that captures semantic meaning and can handle natural language queries.

3. **ReAct Agent Search**: An AI agent using the ReAct (Reasoning and Acting) framework with Qwen2.5-0.5B-Instruct language model that can intelligently choose search strategies, apply filters, and reason about user queries.

The system allows users to filter results by major requirements (CS, Cybersecurity, Data Science), course level (1000-9999), department, and credits. The agent can dynamically decide which search method to use and which filters to apply based on the user's query.

### What is the rationale behind the proposed approach?

The ReAct framework was introduced by Yao et al. (2022) as a way to combine reasoning and acting in language models, showing improved performance on various tasks. Our approach adapts ReAct specifically for educational information retrieval, which differs from previous applications in question-answering or tool use. We combine ReAct with traditional IR methods (TF-IDF) and modern semantic search, creating a hybrid system that leverages the strengths of each approach.

Unlike pure semantic search systems or pure keyword search, our multi-paradigm approach allows users to choose the most appropriate method for their query type. The agent can also intelligently switch between methods, something not typically done in existing course search systems.

**Key Components:**
- FastAPI web application with RESTful API
- PostgreSQL database storing course information
- TF-IDF search implementation with cosine similarity
- Semantic search using pre-computed embeddings
- ReAct agent with language model integration
- Major requirement filtering system
- Web interface with real-time search built with HTML, CSS, and Flowbite

**Results:**

**Limitations:**
- The language model (0.5B parameters) is relatively small and showed limitations in complex reasoning
- Embeddings must be pre-computed, requiring additional storage and computation
- The system requires a PostgreSQL database setup
- Agent search can be slower than direct search methods
- Filter consistency issues were noted in agent mode when filters are enabled

## Approach

### Overall Methodology

The system follows a modular architecture where different search paradigms are implemented as separate modules that can be invoked independently or through the agent. The agent acts as an orchestrator that can reason about queries and select appropriate search strategies.

### Algorithms/Models/Methods

1. **TF-IDF Search**:
   - Tokenization: Extract alphanumeric sequences from course text
   - Stopword removal: Filter common words
   - Term Frequency (TF): Count word occurrences per document
   - Inverse Document Frequency (IDF): Weight terms by rarity across corpus
   - Cosine Similarity: Measure query-document similarity using TF-IDF vectors

2. **Semantic Search**:
   - Model: sentence-transformers/all-mpnet-base-v2 (768-dimensional embeddings)
   - Encoding: Generate embeddings for all courses and queries
   - Similarity: Cosine similarity between query and course embeddings
   - Ranking: Return top-k most similar courses

3. **ReAct Agent**:
   - Model: Qwen/Qwen2.5-0.5B-Instruct
   - Framework: ReAct (Reasoning and Acting) with Thought-Action-Observation loop
   - Tools: keyword_search, semantic_search, finish
   - Prompting: Structured prompts with examples and filter guidance
   - Max Steps: 6 iterations with early stopping after first tool use

### Assumptions and Design Choices

- **Assumptions**:
  - Course descriptions contain sufficient information for semantic matching
  - Users can express their needs in natural language
  - Major requirements can be represented as course code lists

- **Design Choices**:
  - Using a smaller language model (0.5B) for faster inference and lower resource requirements
  - Pre-computing embeddings rather than computing on-the-fly for performance
  - Implementing TF-IDF from scratch for educational purposes and control
  - Using PostgreSQL for structured data storage and querying
  - Allowing users to choose search method or use agent for automatic selection

### Limitations

- **Model Limitations**: The 0.5B parameter model struggles with complex reasoning
- **Embedding Quality**: Semantic search quality depends on the embedding model's understanding of educational domain
- **Scalability**: TF-IDF recomputes IDF for each search, which may be slow for very large corpora
- **Filter Consistency**: Agent may inconsistently apply filters, requiring careful prompt engineering
- **Data Dependency**: System requires structured course data with descriptions, which was a simplified alternative to connecting to SearchNEU and GraduateNU

## Experiments

### Dataset
The dataset consists of course information from Northeastern University's Spring 2026 course catalog. The dataset includes:

- **Total Courses**: [Number of courses]
- **Fields**: subject, number, title, description, min_credits, max_credits, instructors
- **Departments**: [Number of departments]
- **Course Levels**: Distributed across 1000-9999 level buckets
- **Major Requirements**: Pre-defined course lists for CS, Cybersecurity, and Data Science majors

**Data Sources**:
- **Course Information**: Course data was obtained from [SearchNEU](https://github.com/sandboxnu/searchneu), a comprehensive course information platform for Northeastern University. SearchNEU provides structured course data including descriptions, credits, instructors, and scheduling information.
- **Major Requirements**: Major requirement information (course lists for CS, Cybersecurity, and Data Science) was extracted using the course scraper from [GraduateNU](https://github.com/sandboxnu/graduatenu), a Northeastern class scheduler based on degree audit. The scraper extracts major-specific course requirements from the university's course catalog.

The course descriptions vary in length and detail, with some courses having extensive descriptions and others having minimal information.

### Implementation

**Models Used**:
- Qwen/Qwen2.5-0.5B-Instruct for agent reasoning
- sentence-transformers/all-mpnet-base-v2 for embeddings

**Parameters**:
- Agent max steps: 6
- Search result limit (k): 3 (default)
- Embedding dimension: 768
- TF-IDF: Smoothed IDF with (N+1)/(DF+0.5) formula
- Generation config: temperature=0.0, top_k=50, max_new_tokens=128

**Computing Environment**:
- Python 3.10+
- PyTorch for model inference
- PostgreSQL database
- FastAPI web server

### Model Architecture

**ReAct Agent Architecture**:
- Input: User query + optional filters
- Thought Generation: LLM generates reasoning about the query
- Action Selection: LLM selects tool (keyword_search, semantic_search, or finish) with parameters
- Observation: Tool execution returns search results
- Iteration: Process repeats up to max_steps
- Output: Final answer or search results

**Embedding Model**:
- Base: all-mpnet-base-v2 (Microsoft's MPNet)
- Output dimension: 768
- Pre-computed for all courses
- Query-time encoding for user queries

**TF-IDF Implementation**:
- Vocabulary: All unique tokens from corpus (after stopword removal)
- Document vectors: Sparse dictionaries mapping terms to TF-IDF scores
- Similarity: Cosine similarity computed on common terms

## Results

### Main Results

We observed that the TF-IDF search was the most effective and produced the most consistent and reliable outputs, with semantic search also producing accurate but sometimes inconsistent results. The agent struggeld to use the filtering logic to narrow down results, for example: returning a psychology course for a search query of 'object oriented design'

### Supplementary Results

- **Why k=3?**: Provides enough results for comparison without overwhelming users
- **Why 6 max steps?**: Balances agent reasoning capability with response time
- **Why temperature=0.0?**: Ensures deterministic, reproducible agent behavior
- **Why all-mpnet-base-v2?**: Good balance of quality and speed for semantic search
- **Why Qwen 0.5B?**: Fast inference while maintaining reasonable reasoning capability

## Discussion

### Comparison with Existing Approaches

Our implemented tool simplifies the course registration process by removing the manual steps of filtering course results by hand, and instead lets users use natural language queries to get simplified results. This tool therefore takes less time and effort to navigate than Banner or SearchNEU.

### Diagnosis of Issues

For better results from the agentic search, we would likely use a better model with more parameters and refine our prompt enginereing to reduce the inconistencies noted. 

### Future Directions and Improvements

[Speculate on improvements and future work:]

- **Larger Language Models**: Experiment with larger models (1B+, 7B+) for better reasoning
- **Fine-tuning**: Fine-tune the language model on educational domain data
- **Hybrid Search**: Combine TF-IDF and semantic search scores for better results
- **User Feedback Loop**: Incorporate user feedback to improve search quality
- **Multi-turn Conversations**: Support follow-up questions and query refinement
- **Personalization**: Incorporate user history and preferences
- **Explainability**: Provide explanations for why certain courses were recommended
- **Performance Optimization**: Cache embeddings, optimize TF-IDF computation
- **Additional Filters**: Support more filter types (prerequisites, schedule, instructor)
- **Evaluation Metrics**: Develop quantitative metrics for search quality

## Conclusion

We developed a multi-paradigm course search system that successfully combines TF-IDF keyword search, semantic search, and ReAct agent-based search. The system provides a functional web application that allows students to search for courses using natural language queries with intelligent filtering capabilities. While the agent approach shows promise for intelligent query interpretation and tool selection, there are opportunities for improvement in model size, prompt engineering, and result consistency. The project demonstrates the feasibility of applying modern AI agent frameworks to educational information retrieval and provides a foundation for future enhancements.

## Setup

### Prerequisites

- Python 3.10 or higher
- PostgreSQL database
- Git

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Search_NEU_agentic
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   ```

3. **Activate the virtual environment**
   
   On Windows:
   ```bash
   venv\Scripts\activate
   ```
   
   On macOS/Linux:
   ```bash
   source venv/bin/activate
   ```

4. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

5. **Set up environment variables**
   
   Create a `.env` file in the project root:
   ```env
   DATABASE_URL=postgresql://user:password@localhost:5432/search_neu_agentic
   ```

6. **Set up the database**
   
   Create a PostgreSQL database:
   ```sql
   CREATE DATABASE search_neu_agentic;
   ```
   
   Then seed it with course data:
   ```bash
   python data/raw_sp26_course_data/seed.py
   ```

7. **Generate embeddings (optional, for semantic search)**
   ```bash
   python scripts/compute_embeddings.py
   ```
   
   This will create embeddings in `data/embeddings/course_embeddings.parquet`.

8. **Run the application**
   ```bash
   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
   ```

9. **Access the application**
   
   Open your browser and navigate to:
   ```
   http://localhost:8000
   ```

## Project Structure

```
Search_NEU_agentic/
├── app/                    # Main application code
│   ├── agent/             # ReAct agent implementation
│   ├── database/          # Database models and queries
│   ├── llm/               # Language model integration
│   ├── search/            # Search implementations (TF-IDF, embeddings)
│   └── main.py            # FastAPI application entry point
├── data/                  # Data files
│   ├── embeddings/        # Generated embeddings (created by script)
│   └── raw_sp26_course_data/  # Raw course data and seeding script
├── major_requirements/   # Major requirement configurations
├── scripts/               # Utility scripts
├── templates/             # HTML templates
├── tests/                 # Test files
└── requirements.txt       # Python dependencies
```

## Usage

### Search Types

1. **Keyword Search**: Best for specific terms and course codes
2. **Semantic Search**: Best for natural language queries and conceptual searches
3. **Agent Search**: Uses AI to intelligently search and filter based on your query

### Filters

- **Prefix**: Filter by department code (e.g., "CS", "POLS")
- **Class Level**: Filter by course level (1000-9999)
- **Credits**: Filter by minimum credits
- **Major Requirement**: Filter by major requirements (CS, Cybersecurity, Data Science)

## Development

### Running Tests

```bash
pytest tests/
```

### Verifying Embeddings

```bash
python tests/verify_parquet_embeddings.py
```

## Technology Stack

- **Backend**: FastAPI, SQLAlchemy
- **Database**: PostgreSQL
- **ML/AI**: Transformers (Hugging Face), Sentence Transformers, PyTorch
- **Frontend**: HTML, Tailwind CSS, JavaScript
- **Search**: TF-IDF, Cosine Similarity, Semantic Embeddings

## Model Information

- **Language Model**: Qwen/Qwen2.5-0.5B-Instruct (for agent search)
- **Embedding Model**: sentence-transformers/all-mpnet-base-v2 (for semantic search)

## References

[Include all sources used. Use consistent citation style. Examples:]

1. Yao, S., et al. (2022). "ReAct: Synergizing Reasoning and Acting in Language Models." *arXiv preprint arXiv:2210.03629*.

2. Reimers, N., & Gurevych, I. (2019). "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks." *Proceedings of EMNLP-IJCNLP*.

3. Qwen Team. (2024). "Qwen2.5: A Party of Foundation Models." [GitHub Repository](https://github.com/QwenLM/Qwen2.5).

4. Microsoft. (2020). "MPNet: Masked and Permuted Pre-training for Language Understanding." *arXiv preprint arXiv:2004.09297*.

5. FastAPI Documentation. (2024). [https://fastapi.tiangolo.com/](https://fastapi.tiangolo.com/)

6. SQLAlchemy Documentation. (2024). [https://www.sqlalchemy.org/](https://www.sqlalchemy.org/)

7. Sentence Transformers Documentation. (2024). [https://www.sbert.net/](https://www.sbert.net/)

8. SearchNEU. (2024). "The one stop shop for course information at northeastern university." [GitHub Repository](https://github.com/sandboxnu/searchneu). Course information data source.

9. GraduateNU. (2024). "Northeastern class scheduler based on degree audit." [GitHub Repository](https://github.com/sandboxnu/graduatenu). Major requirement data source.
