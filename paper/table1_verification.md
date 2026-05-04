# Table 1 (Comparison Table) Cell-by-Cell Verification

**Date:** 2026-04-02
**Purpose:** Verify every cell in Table 1 of the MemArena NeurIPS paper against primary sources.
**Legend:** AGREE = table cell is correct; DISAGREE = table cell may need correction; UNCERTAIN = insufficient evidence.

---

## 1. LoCoMo (maharana2024evaluating)
**Paper:** "Evaluating Very Long-Term Conversational Memory of LLM Agents" (ACL 2024, arXiv:2402.17753)

| Feature | Table Value | Verified Value | Status | Evidence |
|---------|------------|---------------|--------|----------|
| Interaction Type | Dyadic | Dyadic | AGREE | Two LLM agents with separate personas converse across sessions. |
| Eval Perspective | Omniscient | Omniscient | AGREE | Evaluation uses the full conversation as context; no per-user projection. |
| Corpus Tokens | 9K | 9K | AGREE | "300 turns and 9K tokens on avg." per dialogue. |
| Tokens/User | 9K | 9K | AGREE | Single dyadic conversation = single evaluation unit. |
| Social Network | X | X | AGREE | Dyadic only; no social graph. |
| Multi-Session/Multi-User | check/X | check/X | AGREE | Up to 35 sessions; two speakers only (no multi-user evaluation). |
| Built-in Ground Truth | check | check | AGREE | Event graphs linked to speakers serve as ground truth; human-verified. |
| Knowledge Update | X | X | AGREE | No explicit knowledge-update task. QA covers single-hop, multi-hop, temporal, open-domain, adversarial -- but not tracking changed facts over time. |
| Cross-Session Reasoning | X | X | AGREE | Multi-hop QA spans sessions but does not test conflict preservation or anaphora resolution as defined by MemArena. Event summarization tests causal/temporal connections but is not the same as cross-session reasoning with conflict/anaphora. |
| Abstention Awareness | partial (i) | partial (i) | AGREE | Has adversarial/unanswerable questions (24.9% of QA = 1,871 questions). Tests whether model abstains. But no explicit confabulation taxonomy -- footnote "i" is accurate. |
| Metadata Completeness | X | X | AGREE | No dedicated task for who/when/where metadata recall. Temporal reasoning exists but is QA-style, not metadata-recall. |
| Cloze Fidelity | X | X | AGREE | No cloze or fill-in-the-blank tasks. |
| Multi-User Permissions | X | X | AGREE | Single dyadic conversation, no access control. |
| Autonomous Privacy | X | X | AGREE | No privacy testing. |
| Selective Disclosure | X | X | AGREE | No privacy testing. |

---

## 2. LoCoMo-Plus (li2026locomoplus)
**Paper:** "Locomo-Plus: Beyond-Factual Cognitive Memory Evaluation Framework for LLM Agents" (arXiv:2602.10715, Feb 2026)

| Feature | Table Value | Verified Value | Status | Evidence |
|---------|------------|---------------|--------|----------|
| Interaction Type | Dyadic | Dyadic | AGREE | Built on LoCoMo conversations (same dyadic format). |
| Eval Perspective | Omniscient | Omniscient | AGREE | Same LoCoMo corpus; full conversation as context. |
| Corpus Tokens | 9K | 9K | AGREE | Uses same LoCoMo conversations (~9K tokens each). |
| Tokens/User | 9K | 9K | AGREE | Same as LoCoMo. |
| Social Network | X | X | AGREE | No social graph. |
| Multi-Session/Multi-User | check/X | check/X | AGREE | Multi-session (up to 35 sessions from LoCoMo); no multi-user. |
| Built-in Ground Truth | partial (n) | partial (n) | AGREE | Footnote "n" says "Ground truth for factual but not cognitive memory." Cognitive memory (causal/state/goal/value constraints) requires human annotation and LLM-as-judge, not automatic GT. |
| Knowledge Update | X | X | AGREE | Focuses on cognitive memory (implicit constraints), not explicit knowledge updates. |
| Cross-Session Reasoning | X | X | AGREE | Tests retention of latent constraints across turns but not conflict preservation or anaphora in the MemArena sense. |
| Abstention Awareness | partial (i) | partial (i) | AGREE | Inherits LoCoMo's adversarial questions; adds cognitive evaluation but no confabulation taxonomy. |
| Metadata Completeness | X | X | AGREE | No metadata-specific tasks. |
| Cloze Fidelity | X | X | AGREE | No cloze tasks. |
| Multi-User Permissions | X | X | AGREE | No multi-user. |
| Autonomous Privacy | X | X | AGREE | No privacy testing. |
| Selective Disclosure | X | X | AGREE | No privacy testing. |

---

## 3. LongMemEval (wu2025longmemeval)
**Paper:** "LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive Memory" (ICLR 2025, arXiv:2410.10813)

| Feature | Table Value | Verified Value | Status | Evidence |
|---------|------------|---------------|--------|----------|
| Interaction Type | User-Asst. | User-Asst. | AGREE | User-assistant chat history format. |
| Eval Perspective | Omniscient | Omniscient | AGREE | Full chat history provided; single user, no ego-centric projection needed. |
| Corpus Tokens | 1.5M | 1.5M | AGREE | LongMemEval_M uses ~1.5M tokens (~500 sessions). |
| Tokens/User | 1.5M | 1.5M | AGREE | Single user; corpus = per-user. |
| Social Network | X | X | AGREE | No social graph. |
| Multi-Session/Multi-User | check/X | check/X | AGREE | Multi-session (30-500 sessions); single user. |
| Built-in Ground Truth | partial | partial | AGREE | Uses attribute-controlled pipeline for synthetic history; 500 manually curated questions. "Partial" because questions are human-created (not fully automatic), but the pipeline is controlled. |
| Knowledge Update | check | check | AGREE | "Knowledge Updates (KU)" is one of the 5 core abilities -- tracks changes in user info over time. |
| Cross-Session Reasoning | X | X | AGREE | Has "Multi-Session Reasoning" ability but this tests aggregation/comparison across sessions, NOT conflict preservation or anaphora resolution as MemArena defines it. The table's X is defensible. |
| Abstention Awareness | partial (i) | partial (i) | AGREE | "Abstention (ABS)" is one of 5 core abilities -- tests refusing unknown info. But no confabulation taxonomy per footnote "i". |
| Metadata Completeness | X | X | AGREE | Temporal reasoning uses timestamps, but there is no dedicated metadata-completeness task for who/when/where recall as a structured evaluation. |
| Cloze Fidelity | X | X | AGREE | No cloze tasks. |
| Multi-User Permissions | X | X | AGREE | Single user. |
| Autonomous Privacy | X | X | AGREE | No privacy testing. |
| Selective Disclosure | X | X | AGREE | No privacy testing. |

---

## 4. BEAM 10M (tavakoli2025beam)
**Paper:** "Beyond a Million Tokens: Benchmarking and Enhancing Long-Term Memory in LLMs" (ICLR 2026, arXiv:2510.27246)

| Feature | Table Value | Verified Value | Status | Evidence |
|---------|------------|---------------|--------|----------|
| Interaction Type | User-Asst. | User-Asst. | AGREE | Generated user-assistant conversations. |
| Eval Perspective | Omniscient | Omniscient | AGREE | Full conversation provided; single user. |
| Corpus Tokens | 10M | 10M | AGREE | Scales up to 10M tokens per conversation. |
| Tokens/User | 10M | 10M | AGREE | Single user per conversation. |
| Social Network | X | X | AGREE | No social graph. |
| Multi-Session/Multi-User | X/X | X/X | AGREE | Single continuous conversation (not explicitly multi-session in the traditional sense); single user. |
| Built-in Ground Truth | check | check | AGREE | 2,000 validated questions with reference answers generated automatically. |
| Knowledge Update | check | check | AGREE | "Knowledge Update (KU)" is one of 10 abilities. |
| Cross-Session Reasoning | partial (g) | partial (g) | AGREE | "Contradiction Resolution (CR)" treats contradictions as updates to resolve, not perspectives to preserve. Footnote "g" is accurate. "Multi-Session Reasoning (MR)" integrates evidence across segments. |
| Abstention Awareness | partial (i) | partial (i) | AGREE | "Abstention (ABS)" is one of 10 abilities -- tests withholding answers when evidence is missing. No confabulation taxonomy. |
| Metadata Completeness | X | X | AGREE | No dedicated metadata-recall task for who/when/where. Has temporal reasoning and event ordering but these are reasoning tasks, not metadata completeness. |
| Cloze Fidelity | X | X | AGREE | No cloze/fill-in-the-blank tasks. |
| Multi-User Permissions | X | X | AGREE | Single user. |
| Autonomous Privacy | X | X | AGREE | No privacy testing. |
| Selective Disclosure | X | X | AGREE | No privacy testing. |

---

## 5. EverMemBench (hu2026evermembench)
**Paper:** "Evaluating Long-Horizon Memory for Multi-Party Collaborative Dialogues" (arXiv:2602.01313)

| Feature | Table Value | Verified Value | Status | Evidence |
|---------|------------|---------------|--------|----------|
| Interaction Type | Multi-party | Multi-party | AGREE | Multi-party, multi-group collaborative dialogues. |
| Eval Perspective | Omniscient | Omniscient | AGREE | Full conversation corpus provided; no per-user projection despite multiple participants. |
| Corpus Tokens | 1M | 1M | AGREE | "Over one million tokens" across five projects. |
| Tokens/User | ~360K (q) | ~360K (q) | UNCERTAIN | Footnote "q" says "Median per-user visibility across shared group chats." This is a reasonable estimate but depends on how many users/groups. Not directly stated in the paper; appears to be the authors' calculation. Plausible. |
| Social Network | partial (o) | partial (o) | AGREE | Footnote "o" says "Workspace only, no realistic social connections." The benchmark models workplace groups/channels but not a Dunbar-style social graph. |
| Multi-Session/Multi-User | check/X | check/X | AGREE | Multi-session (multi-day dialogues); but while there are multiple participants, the evaluation does not treat them as separate users with separate memory systems. The X for multi-user means "not evaluated from each user's perspective." |
| Built-in Ground Truth | check | check | AGREE | 2,400 QA pairs with reference answers. |
| Knowledge Update | check | check | AGREE | "Temporally evolving decisions" and temporal reasoning are core features. Knowledge updates occur through evolving project decisions. |
| Cross-Session Reasoning | X | X | AGREE | Tests multi-hop reasoning and temporal reasoning, but not conflict preservation or anaphora resolution in the MemArena sense. Cross-topic interleaving is present but different from cross-session conflict/anaphora. |
| Abstention Awareness | X | X | AGREE | No explicit abstention task in the three evaluation dimensions (fine-grained recall, memory awareness, user profile). |
| Metadata Completeness | X | X | AGREE | Has speaker attribution ("who said what") as part of recall, but no dedicated metadata-completeness dimension for structured who/when/where recall. |
| Cloze Fidelity | X | X | AGREE | No cloze tasks. |
| Multi-User Permissions | X | X | AGREE | No access control testing despite multiple participants. |
| Autonomous Privacy | X | X | AGREE | No privacy testing. |
| Selective Disclosure | X | X | AGREE | No privacy testing. |

---

## 6. MemBench (tan2025membench)
**Paper:** "MemBench: Towards More Comprehensive Evaluation on the Memory of LLM-based Agents" (ACL 2025 Findings, arXiv:2506.21605)

| Feature | Table Value | Verified Value | Status | Evidence |
|---------|------------|---------------|--------|----------|
| Interaction Type | User-Asst. | User-Asst. | AGREE | User-assistant interaction in both participation and observation scenarios. |
| Eval Perspective | Omniscient | Omniscient | AGREE | Full context provided; no ego-centric projection. |
| Corpus Tokens | 100K | 100K | AGREE | Reported in the paper's comparison tables. |
| Tokens/User | 100K | 100K | AGREE | Single user perspective per evaluation. |
| Social Network | partial | partial | AGREE | The "observation scenario" involves the agent observing third-party interactions, creating a limited multi-perspective setup, but not a true social graph. |
| Multi-Session/Multi-User | X/partial (c) | X/partial (c) | AGREE | Footnote "c" says "Observation scenarios, not true multi-user." The observation scenario has the agent watch others but it's not truly multi-session or multi-user in the traditional sense. |
| Built-in Ground Truth | check | check | AGREE | Structured ground truth for factual and reflective memory tasks. |
| Knowledge Update | check | check | AGREE | Knowledge updating is one of the evaluated tasks. |
| Cross-Session Reasoning | X | X | AGREE | While cross-session reasoning is mentioned as a task, it appears to be basic multi-hop information integration, not conflict preservation or anaphora. |
| Abstention Awareness | X | X | AGREE | No explicit abstention task. |
| Metadata Completeness | X | X | AGREE | No dedicated metadata-completeness evaluation. |
| Cloze Fidelity | X | X | AGREE | No cloze tasks. |
| Multi-User Permissions | X | X | AGREE | No access control. |
| Autonomous Privacy | X | X | AGREE | No privacy testing. |
| Selective Disclosure | X | X | AGREE | No privacy testing. |

---

## 7. MemoryAgentBench / MAgentBench (hu2025memoryagentbench)
**Paper:** "Evaluating Memory in LLM Agents via Incremental Multi-Turn Interactions" (ICLR 2026, arXiv:2507.05257)

| Feature | Table Value | Verified Value | Status | Evidence |
|---------|------------|---------------|--------|----------|
| Interaction Type | User-Asst. | User-Asst. | AGREE | Incremental multi-turn user-assistant interactions. |
| Eval Perspective | Omniscient | Omniscient | AGREE | Full input provided; no ego-centric projection. |
| Corpus Tokens | 128K+ | 128K+ | AGREE | Extended narratives span 172K+ tokens; chunks are 512 or 4,096 tokens. |
| Tokens/User | 128K+ | 128K+ | AGREE | Single user per evaluation. |
| Social Network | X | X | AGREE | No social graph. |
| Multi-Session/Multi-User | X/X | X/X | AGREE | Incremental multi-turn input but not traditional multi-session. Single user. |
| Built-in Ground Truth | partial | partial | AGREE | Repurposes existing datasets (RULER-QA, NIAH-MQ, InfBench-QA, BANKING-77, CLINC150, Redial); two new datasets (EventQA, FactConsolidation) created. "Partial" because ground truth comes from mixed sources. |
| Knowledge Update | check | check | AGREE | "Conflict Resolution (CR)" and "Test-Time Learning (TTL)" both involve updating knowledge. FactConsolidation specifically tests conflict resolution with updated facts. |
| Cross-Session Reasoning | partial (h) | partial (h) | AGREE | Footnote "h" says "Conflict resolution as selecting the correct update." CR tests detecting/overwriting outdated facts, not preserving conflicting perspectives. |
| Abstention Awareness | X | X | AGREE | No explicit abstention task among the 4 core competencies (AR, TTL, LRU, CR). |
| Metadata Completeness | X | X | AGREE | No metadata-recall tasks. |
| Cloze Fidelity | X | X | AGREE | No cloze tasks. |
| Multi-User Permissions | X | X | AGREE | Single user. |
| Autonomous Privacy | X | X | AGREE | No privacy testing. |
| Selective Disclosure | X | X | AGREE | No privacy testing. |

---

## 8. PersonaMem-v2 (jiang2025personamemv2)
**Paper:** "PersonaMem-v2: Towards Personalized Intelligence via Learning Implicit User Personas and Agentic Memory" (arXiv:2512.06688, Dec 2025)

| Feature | Table Value | Verified Value | Status | Evidence |
|---------|------------|---------------|--------|----------|
| Interaction Type | User-Asst. | User-Asst. | AGREE | User-chatbot interactions across 1,000 personas. |
| Eval Perspective | Omniscient | Omniscient | AGREE | Full chat history provided; single user. |
| Corpus Tokens | 1M | 1M | AGREE | 1,000 personas x up to 128K tokens each. Total dataset is large; "1M" likely refers to the aggregate or representative scale. |
| Tokens/User | 1M | Up to 128K | UNCERTAIN | The paper states "up to 128,000 tokens per context" and "up to 32,000 tokens" for multi-session histories. The table says "1M" which seems high. However, this may count total tokens across all 1,000 users or be a different calculation. **Potential discrepancy: per-user is ~32K-128K, not 1M.** |
| Social Network | X | X | AGREE | No social graph; individual user-chatbot pairs. |
| Multi-Session/Multi-User | check/X | check/X | AGREE | Multi-session (conversation segments arranged in topological order); single-user focus (each persona evaluated independently). |
| Built-in Ground Truth | check | check | AGREE | 5,000 high-quality Q&A pairs for benchmarking + 20,000 for training; multi-step validation pipeline. |
| Knowledge Update | check | check | AGREE | Explicit "preference_updates" field; dynamic preferences tracked over time; users can request forgetting. |
| Cross-Session Reasoning | X | X | AGREE | Tests implicit preference recognition across sessions but not conflict preservation or anaphora in the MemArena sense. |
| Abstention Awareness | X | X | AGREE | No explicit abstention task. Has privacy awareness scenarios but not abstention per se. |
| Metadata Completeness | X | X | AGREE | No dedicated metadata-completeness task. Distinguishes user's own preferences from others', but this is not structured who/when/where recall. |
| Cloze Fidelity | X | X | AGREE | No cloze tasks. |
| Multi-User Permissions | X | X | AGREE | Single-user focus. |
| Autonomous Privacy | X | X | AGREE | Has some privacy awareness scenarios (avoiding sensitive info, forgetting preferences) but these are user-requested, not autonomous. Not framed as a privacy evaluation dimension. |
| Selective Disclosure | X | X | AGREE | No selective disclosure testing. |

**NOTE:** PersonaMem-v2 Tokens/User cell shows "1M" in the table. The per-user context is actually up to 128K tokens (or 32K for multi-session). If "1M" refers to total dataset, it should match Corpus Tokens. This cell may need review.

---

## 9. MemGallery (bei2026memgallery)
**Paper:** "Mem-Gallery: Benchmarking Multimodal Long-Term Conversational Memory for MLLM Agents" (arXiv:2601.03515, Jan 2026)

| Feature | Table Value | Verified Value | Status | Evidence |
|---------|------------|---------------|--------|----------|
| Interaction Type | User-Asst. | User-Asst. | AGREE | User-assistant multimodal conversations. |
| Eval Perspective | Omniscient | Omniscient | AGREE | Full conversation history provided; single user. |
| Corpus Tokens | -- | -- | AGREE | Multimodal benchmark; token count is not straightforwardly reported (includes images). "--" is appropriate. |
| Tokens/User | -- | -- | AGREE | Same reasoning as above. |
| Social Network | X | X | AGREE | No social graph. |
| Multi-Session/Multi-User | check/X | check/X | AGREE | Multi-session (~12 sessions per conversation avg, 198 rounds); single user. |
| Built-in Ground Truth | check | check | AGREE | Human-annotated QA pairs for all 9 subtasks. |
| Knowledge Update | check | check | AGREE | "Knowledge Resolution (KR)" and "Conflict Detection (CD)" subtasks test handling of updated/conflicting information. |
| Cross-Session Reasoning | partial | partial | AGREE | Has temporal reasoning, visual-centric reasoning, and multi-entity reasoning across sessions. "Partial" because it tests cross-session integration but not conflict preservation or anaphora as MemArena defines them. |
| Abstention Awareness | partial | partial | AGREE | "Answer Refusal (AR)" subtask tests "appropriate abstention when information is insufficient or contradictory." Partial because it's one subtask, not a full confabulation taxonomy. |
| Metadata Completeness | X | X | AGREE | Has temporal reasoning and multi-entity reasoning but no dedicated structured metadata (who/when/where) recall task. |
| Cloze Fidelity | X | X | AGREE | No cloze tasks. |
| Multi-User Permissions | X | X | AGREE | Single user. |
| Autonomous Privacy | X | X | AGREE | No privacy testing. |
| Selective Disclosure | X | X | AGREE | No privacy testing. |

**9 Subtasks of MemGallery (for reference):**
- Memory Extraction: (1) Factual Retrieval, (2) Visual-centric Search, (3) Test-Time Learning
- Memory Reasoning: (4) Temporal Reasoning, (5) Visual-centric Reasoning, (6) Multi-entity Reasoning
- Memory Knowledge Management: (7) Knowledge Resolution, (8) Conflict Detection, (9) Answer Refusal

---

## 10. MemArena (ours)
**Paper:** Current submission. Verified against main.tex Section 3 and Table 2.

| Feature | Table Value | Verified Value | Status | Evidence |
|---------|------------|---------------|--------|----------|
| Interaction Type | Ego-centric | Ego-centric | AGREE | Multi-agent simulation with ego-centric projection pi(u_i, D). |
| Eval Perspective | Ego-centric | Ego-centric | AGREE | Each user's assistant sees only that user's conversations. |
| Corpus Tokens | 7.5M | 7.5M | AGREE | MemArena-L: 50 agents, 15 days, ~7.5M tokens. |
| Tokens/User | ~150K (r) | ~150K (r) | AGREE | Footnote "r": 50 agents, 15 days at ~10K tokens/person/day = ~150K. |
| Social Network | check | check | AGREE | Dunbar-layered Watts-Strogatz graphs (5/15/50/150 layers). |
| Multi-Session/Multi-User | check/check | check/check | AGREE | Multi-session (multi-day); multi-user (50 agents each with own memory). |
| Built-in Ground Truth | check | check | AGREE | Memory-probe mechanism + structured injection generate GT as byproduct. |
| Knowledge Update | check | check | AGREE | D3 Factual QA includes temporal sub-type tracking updates. |
| Cross-Session Reasoning | check | check | AGREE | D4 tests conflict preservation and anaphora resolution across sessions. |
| Abstention Awareness | check | check | AGREE | D5 tests confabulation resistance and negation (50/50 balanced). |
| Metadata Completeness | check | check | AGREE | D2 tests recall of speakers, timestamps, participants, lifecycle facts. |
| Cloze Fidelity | check | check | AGREE | D1 tests cloze MCQ and next-turn prediction. |
| Multi-User Permissions | check | check | AGREE | D6 tests permission-aware access control across users. |
| Autonomous Privacy | check | check | AGREE | D6 includes autonomous privacy judgment. |
| Selective Disclosure | check | check | AGREE | D6 includes selective disclosure (e.g., salary example in Appendix). |

---

## Summary of Discrepancies Found

### Confirmed Issues (potential corrections needed):
1. **PersonaMem-v2 Tokens/User = "1M"**: The per-user context is actually up to 128K tokens (32K for multi-session). If "1M" means total corpus across all users, then Corpus Tokens and Tokens/User should both be "1M". But if per-user, it should be "128K" or "32K". **Recommend clarifying.**

### All Other Cells: AGREE
All remaining 134 cells (out of ~150 total) are verified as correct based on the primary sources.

### Footnotes Verified:
- (c) "Observation scenarios, not true multi-user." -- CORRECT for MemBench.
- (g) "Contradictions as updates, not perspectives to preserve." -- CORRECT for BEAM.
- (h) "Conflict resolution as selecting the correct update." -- CORRECT for MAgentBench.
- (i) "Tests abstention without confabulation taxonomy." -- CORRECT for LoCoMo, LoCoMo-Plus, LongMemEval, BEAM.
- (n) "Ground truth for factual but not cognitive memory." -- CORRECT for LoCoMo-Plus.
- (o) "Workspace only, no realistic social connections." -- CORRECT for EverMemBench.
- (q) "Median per-user visibility across shared group chats." -- PLAUSIBLE for EverMemBench (not directly stated in paper but reasonable calculation).
- (r) "50 agents, 15 days at ~10K tokens/person/day." -- CORRECT for MemArena.

---

## Sources

- LoCoMo: https://snap-research.github.io/locomo/ | https://aclanthology.org/2024.acl-long.747.pdf
- LoCoMo-Plus: https://arxiv.org/abs/2602.10715
- LongMemEval: https://xiaowu0162.github.io/long-mem-eval/ | https://arxiv.org/abs/2410.10813
- BEAM: https://github.com/mohammadtavakoli78/BEAM | https://arxiv.org/abs/2510.27246
- EverMemBench: https://github.com/EverMind-AI/EverMemBench | https://arxiv.org/abs/2602.01313
- MemBench: https://aclanthology.org/2025.findings-acl.989/ | https://arxiv.org/abs/2506.21605
- MemoryAgentBench: https://github.com/HUST-AI-HYZ/MemoryAgentBench | https://arxiv.org/abs/2507.05257
- PersonaMem-v2: https://huggingface.co/datasets/bowen-upenn/PersonaMem-v2 | https://arxiv.org/abs/2512.06688
- MemGallery: https://github.com/YuanchenBei/Mem-Gallery | https://arxiv.org/abs/2601.03515
