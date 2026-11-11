from __future__ import annotations

STATIC_PREFIX = """You answer questions about three research papers using ONLY the numbered passages below.
Rules:
1. Use only facts stated in the passages. Do not use what you remember about these papers: this library pins specific versions and your memory may be of another one.
2. Write at most 4 short sentences, each on its own line.
3. End every line with the id(s) of the passage(s) that support it in square brackets, e.g. [M12] or [A3, A7].
4. Copy numbers, names and units exactly as they appear in the passage.
5. If the passages do not state the answer, output exactly NOT_IN_PAPERS on the first line and nothing else.
6. If the question names no paper and the passages give different answers for different papers, write one line per paper and name the paper in each line.
7. A passage describing someone else's system (a baseline, related work) is not a fact about the paper's own model. Say whose result a number is.

Example
PASSAGES
[X1] (Example Paper | §2 Method) The model uses 6 layers and a hidden size of 512.
[X2] (Example Paper | §4 Results | Table 1) Model | Score
Ours | 28.4
Baseline-B | 25.1
QUESTION: What score does the model reach, and what does the baseline reach?
ANSWER
The paper's own model reaches a score of 28.4 [X2]
Baseline-B reaches 25.1 [X2]

Example
PASSAGES
[X1] (Example Paper | §2 Method) The model uses 6 layers and a hidden size of 512.
QUESTION: Which optimizer was used?
ANSWER
NOT_IN_PAPERS
"""


def build_draft_messages(question: str, hits) -> list[dict]:
    passages = "\n".join(f"[{h.chunk.id}] ({h.chunk.header.strip('[]')}) {h.chunk.text}" for h in hits)
    return [{"role": "user", "content": f"{STATIC_PREFIX}\nPASSAGES\n{passages}\nQUESTION: {question}\nANSWER\n"}]
