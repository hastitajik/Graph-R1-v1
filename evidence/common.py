"""
Shared constants and settings for the evidence pipeline. Light imports only.

Endpoints are read from the environment so that no internal address lives in the code:

  EVIDENCE_LLM_BASE_URL  OpenAI-compatible server of the frozen LLM   (default http://localhost:8000/v1)
  EVIDENCE_LLM_MODEL     model name served there                        (default Qwen3.8-27B)
  EVIDENCE_SEARCH_URL    the Graph-R1 /search API, e.g. http://<ip>:8001/search   (no default)
  EVIDENCE_DATA_DIR      folder with graph_chunk_entity_relation.graphml and the chunk store
                         (default expr/2WikiMultiHopQA)

Every command-line script also accepts the matching flag, which wins over the variable.
"""
import os
import re

# ?D, ?M, ?T1 ... : the unknowns of an obligation text
VAR = re.compile(r"\?[A-Za-z]\w*")

LLM_BASE_URL = os.environ.get("EVIDENCE_LLM_BASE_URL", "http://localhost:8000/v1")
LLM_MODEL = os.environ.get("EVIDENCE_LLM_MODEL", "Qwen3.8-27B")
SEARCH_URL = os.environ.get("EVIDENCE_SEARCH_URL", "")
DATA_DIR = os.environ.get("EVIDENCE_DATA_DIR", "expr/2WikiMultiHopQA")
