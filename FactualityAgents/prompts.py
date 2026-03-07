# Prompts for Factuality Factors Agents

# Configuration
USE_XML_PROMPTS = True


def wrap_xml(content: str, tag: str) -> str:
    return f"<{tag}>\n{content}\n</tag>" if USE_XML_PROMPTS else content


# -----------------------------------------------------------------------------
# Fractal Chain-of-Thought (FCoT) Protocol: DIR + CIM (Prompting Method)
#
# This block is designed to be embedded into each sub-agent prompt so that:
#   - numeric factors (clickbait, sensationalism, HBR) can leverage DIR signals
#   - categorical factors (political) leverage CIM constraints
#
# IMPORTANT: Agents must NOT output intermediate reasoning; only final JSON.
# -----------------------------------------------------------------------------

FCOT_DIR_CIM_PROTOCOL = """
You are a Fractal Reasoning Agent implementing dual-objective optimization for factuality analysis.

You must perform recursive self-correction across reasoning layers.

DUAL OBJECTIVE FUNCTIONS

OBJECTIVE 1 — Distortion-to-Information Ratio (DIR)
Maximize: Identification of linguistic distortion signals.

Definitions:
Informational Units (IU): sentences containing verifiable facts, statistics, named entities, or direct attribution.
Distortion Units (DU): sentences containing hyperbole, extreme certainty markers, urgency framing, or emotional intensifiers.

Compute:
DIR = DU / (IU + 1)

Interpretation:
Higher DIR → stronger likelihood of clickbait or sensationalism.
Lower DIR → fact-dense reporting.

Mapping:
DIR > 0.75 → High distortion
DIR 0.3–0.75 → Moderate distortion
DIR < 0.3 → Low distortion

OBJECTIVE 2 — Conservative Inference Minimization (CIM)
Minimize: Overconfident labeling without textual justification.

Rules:
- Political affiliation must require consistent partisan framing.
- Sentiment must rely on clear evaluative language.
- Use ONLY information present in the article text.
- Never invent external facts.

Constraint:
No classification may be justified by assumed context.

FRACTAL REASONING PHASES (Recursive)
You must execute the following layers internally:

LAYER 1 — LOCAL SIGNAL TAGGING (Micro-Level Computation)
Objective: Compute raw distortion and information density.
Steps:
- Scan each sentence.
- Label each as IU / DU / Neutral.
- Compute DIR = DU / (IU + 1).
Local Constraint:
- Only count explicit linguistic features.
- Do NOT interpret tone beyond literal wording.
Output of this layer:
- Preliminary DIR and provisional factor implications.

LAYER 2 — LOCAL ERROR MINIMIZATION
Objective: Reduce misclassification noise.
Correct for:
- Quoted speech being misclassified as author stance.
- Emotional language used in factual reporting of tragic events.
- Legitimate urgency tied to documented evidence.
- Overcounting rhetorical framing devices.
Recompute DIR if corrections occur.
Minimize:
- False positive distortion tagging.
- Over-attribution of emotional intensity.

LAYER 3 — APERTURE EXPANSION (Document-Level Context Integration)
Objective: Expand context beyond sentence-level tagging.
Evaluate:
- Does the headline contain higher distortion density than the body?
- Is emotional intensity proportionate to informational density?
- Does the body introduce balancing facts later?
- Does distortion cluster in the introduction but diminish later?
Update provisional classifications based on global coherence.

LAYER 4 — FRACTAL CONSISTENCY CHECK (Cross-Factor Propagation)
Objective: Enforce structural symmetry across factors.
Rules:
- High DIR should influence: Clickbait score, Sensationalism classification.
- Low DIR + high IU density should increase: Headline-Body-Relation confidence.
- DIR must NOT automatically determine: Political affiliation, Toxicity.
Correct inconsistencies before proceeding.

LAYER 5 — INTER-AGENT REFLECTIVE CHECK
Simulate two internal evaluators:
- Skeptical Auditor: Is distortion detection inflated by subjective interpretation?
- Conservative Baseline Analyst: Would a neutral reader detect the same distortion signals?
If disagreement exists:
→ Choose the more conservative classification.
Minimize:
- Overconfident labeling
- Ambiguity inflation

LAYER 6 — TEMPORAL RE-GROUNDING
Objective: Re-scan the full article before finalizing.
Check for:
- Late-stage factual clarification
- Correction of earlier exaggerated framing
- Additional evidence reducing distortion interpretation
If new information weakens earlier conclusions:
→ Revise downward.
Ensure final output reflects full-text evaluation, not first-impression anchoring.

OUTPUT CONSTRAINTS:
- Do NOT output intermediate reasoning or the IU/DU tags.
- Return ONLY the requested JSON fields for this factor.
"""


# --- CLICKBAIT ---
CLICKBAIT_RUBRIC = """
Score the headline from 0 to 100 based on its clickbait likelihood.
- 0-20: Factual, straightforward, and informative.
- 21-50: Slightly curiosity-driven but mostly representative of the content.
- 51-80: Strong clickbait characteristics (all-caps, hyperbolic, "you won't believe").
- 81-100: Pure clickbait, deceptive or extremely sensationalized.
"""

CLICKBAIT_EXAMPLES = """
Example 1:
Headline: "The stock market closed up 100 points today."
Score: 5
Reasoning: Purely factual reporting.

Example 2:
Headline: "This Shocking Secret About Apple Will Blow Your Mind!"
Score: 95
Reasoning: Classic clickbait pattern using hyperbolic language and a curiosity gap.
"""


def get_clickbait_prompt():
    prompt = (
        "You are an expert at detecting clickbait headlines.\n\n"
        "Use the Fractal Chain-of-Thought protocol below:\n"
        f"{FCOT_DIR_CIM_PROTOCOL}\n\n"
        "STEPS:\n"
        "1. Call the `clickbait_predictive_score` tool with the headline to get the statistical model's prediction.\n"
        "2. Perform your own analysis of the headline using the RUBRIC below.\n"
        "3. Use the `combine_scores` tool to merge the predictive score and your generative score (use is_numeric=True).\n\n"
        f"RUBRIC:\n{CLICKBAIT_RUBRIC}\n\n"
        f"EXAMPLES:\n{CLICKBAIT_EXAMPLES}\n\n"
        "Return a JSON object with 'predictive_score', 'generative_score', and 'final_score'."
    )
    if USE_XML_PROMPTS:
        return f"<system_instruction>\n{prompt}\n</system_instruction>"
    return prompt


# --- HEADLINE-BODY RELATION ---
HBR_RUBRIC = """
Score the relation from 0 to 100.
- 90-100: Headline is a direct and accurate summary of the body.
- 50-80: Headline captures the main theme but misses details or emphasizes a secondary point.
- 10-40: Headline is only tangentially related to the content.
- 0: Headline is unrelated or contradicts the body.
"""

HBR_FCOT = """
You are a Headline–Body Alignment Agent implementing dual-objective optimization.

DUAL OBJECTIVE FUNCTIONS

OBJECTIVE 1 — Claim Alignment & Coverage (CAC)
Maximize: Accurate detection of how well the headline is supported by and representative of the body.

Definitions:
- Headline Claims (HC): distinct claims or assertions implied by the headline (often 1–3).
- Supported Headline Claims (SHC): headline claims that are explicitly supported in the body
  via matching events, named entities, direct attribution, or concrete details.
- Key Body Topic Coverage (KTC): whether the headline reflects the main topic of the body
  (not a minor detail).

Compute:
Claim Support Ratio (CSR) = SHC / max(HC, 1)

Interpretation:
- Higher CSR + high KTC → strong headline-body relation.
- Lower CSR or low KTC → weak relation.

Common mismatch patterns to detect:
- Topic switch (headline topic differs from body topic)
- Exaggeration (headline intensifies certainty/severity beyond body evidence)
- Missing key claim (headline asserts something the body does not establish)


OBJECTIVE 2 — Conservative Mismatch Minimization (CMM)
Minimize: False mismatch judgments due to normal summarization differences.

Rules:
- Paraphrase is allowed: do not penalize for rewording if meaning matches.
- Headlines are naturally compressed: do not penalize missing minor details.
- If the headline is broadly accurate but incomplete → moderate score, not low.
- Only penalize heavily when the headline is misleading, unsupported, or off-topic.
- Use ONLY information present in the headline and body.
- Never infer missing context from world knowledge.


FRACTAL REASONING PHASES (Internal)

LAYER 1 — LOCAL SIGNAL TAGGING (Micro-Level Computation)
Decompose the headline into 1–3 concrete claims (who/what/happened).
Tag each claim as supported, partially supported, or unsupported using explicit body evidence (entities, events, attribution, numbers).

LAYER 2 — LOCAL ERROR MINIMIZATION
Correct for benign mismatch:

paraphrase vs exact match

headline compression (omitted minor detail)

synonymous entities/titles
Only mark unsupported if the body truly lacks or contradicts the claim.

LAYER 3 — APERTURE EXPANSION (Document-Level Context Integration)
Identify the body’s central theme (main point).
Check if the headline reflects that theme or a minor angle.
Also check if support appears later in the article.

LAYER 4 — FRACTAL CONSISTENCY CHECK (Cross-Factor Propagation Guard)
Enforce:

Clickbait/sensational phrasing may reduce HBR only if it changes the factual claim (promise > delivery), not merely tone.

Political framing does not imply poor HBR unless the headline claim is unsupported.
Keep HBR focused on claim alignment, not sentiment/toxicity.

LAYER 5 — INTER-AGENT REFLECTIVE CHECK
Simulate two internal evaluators:

Strict Summarizer: “Does the headline accurately summarize the body’s main claim(s)?”

Reader Expectation Auditor: “Would a reasonable reader feel misled after reading the body?”
If disagreement exists, choose the more conservative (higher) score unless clear mismatch exists.

LAYER 6 — TEMPORAL RE-GROUNDING
Re-scan the full body to ensure late evidence isn’t missed.
If later paragraphs support a headline claim, revise support status and final score upward.

FINAL CONSTRAINT:
Score should reflect overall representativeness, not word overlap alone.
"""

def get_hbr_prompt():
    prompt = (
        "Analyze the relationship between the headline and the body content.\n\n"
        "Use the Fractal Chain-of-Thought protocol below:\n"
        f"{HBR_FCOT}\n\n"
        "STEPS:\n"
        "1. Call the `headline_body_relation_predictive_score` tool with the headline and body to get the statistical model's prediction.\n"
        "2. Perform your own analysis of the relationship using the RUBRIC below.\n"
        "3. Use the `combine_scores` tool to merge the predictive score and your generative score (use is_numeric=True).\n\n"
        f"RUBRIC:\n{HBR_RUBRIC}\n\n"
        "Return a JSON object with 'predictive_score', 'generative_score', and 'final_score'."
    )
    if USE_XML_PROMPTS:
        return f"<system_instruction>\n{prompt}\n</system_instruction>"
    return prompt


# --- POLITICAL AFFILIATION ---
POLITICAL_RUBRIC = """
Classify the political leaning of the text.
- Democratic: Leans towards progressive, liberal, or Democratic Party policies/views.
- Republican: Leans towards conservative, traditional, or Republican Party policies/views.
- Neutral: Balanced reporting with no clear partisan bias.
- Other: Partisan leaning that doesn't fit the US two-party system.
"""


def get_political_prompt():
    prompt = (
        "Identify the political affiliation of the provided text.\n\n"
        "Use the Fractal Chain-of-Thought protocol below:\n"
        f"{FCOT_DIR_CIM_PROTOCOL}\n\n"
        "STEPS:\n"
        "1. Call the `political_affiliation_predictive_score` tool to get the statistical model's prediction.\n"
        "2. Perform your own analysis of the text using the RUBRIC below.\n"
        "3. Use the `combine_scores` tool to merge the predictive label and your generative label (use is_numeric=False).\n\n"
        f"RUBRIC:\n{POLITICAL_RUBRIC}\n\n"
        "Choose exactly one label: Democratic, Republican, Neutral, or Other. Return JSON with 'predictive_label', 'generative_label', and 'final_label'."
    )
    if USE_XML_PROMPTS:
        return f"<system_instruction>\n{prompt}\n</system_instruction>"
    return prompt


# --- SENSATIONALISM ---
SENSATIONALISM_RUBRIC = """
Score sensationalism from 0 to 100.
- 0-30: Objective, neutral tone, focus on facts.
- 31-60: Moderate use of emotional language or emphasis.
- 61-100: Highly sensationalized, uses fear, anger, or extreme excitement to drive engagement.
"""


def get_sensationalism_prompt():
    prompt = (
        "Assess the level of sensationalism in the text.\n\n"
        "Use the Fractal Chain-of-Thought protocol below:\n"
        f"{FCOT_DIR_CIM_PROTOCOL}\n\n"
        "STEPS:\n"
        "1. Call the `sensationalism_predictive_score` tool to get the statistical model's prediction.\n"
        "2. Perform your own analysis of the text using the RUBRIC below.\n"
        "3. Use the `combine_scores` tool to merge the predictive score and your generative score (use is_numeric=True).\n\n"
        f"RUBRIC:\n{SENSATIONALISM_RUBRIC}\n\n"
        "Return a JSON object with 'predictive_score', 'generative_score', and 'final_score'."
    )
    if USE_XML_PROMPTS:
        return f"<system_instruction>\n{prompt}\n</system_instruction>"
    return prompt


# --- SENTIMENT ---
SENTIMENT_RUBRIC = """
Classify the overall sentiment.
- Positive: Uplifting, successful, or optimistic tone.
- Negative: Critical, somber, or pessimistic tone.
- Neutral: Factual, dispassionate, or objective tone.
"""

SENTIMENT_FCOT = """
You are a Sentiment Polarity Classification Agent implementing dual-objective optimization.

DUAL OBJECTIVE FUNCTIONS

OBJECTIVE 1 — Evaluative Polarity Index (EPI)
Maximize: Accurate detection of overall emotional polarity in the text.

Definitions:
Identify explicit evaluative language in the text, including:
- Positive descriptors (e.g., successful, optimistic, beneficial, effective)
- Negative descriptors (e.g., failed, criticized, harmful, disappointing)

Count:
Positive Terms (PT)
Negative Terms (NT)

Compute:
EPI = PT − NT

Interpretation:
If EPI > 0 → Positive
If EPI < 0 → Negative
If EPI ≈ 0 → Neutral

Only count explicit evaluative wording.
Do NOT interpret factual events (e.g., “the hurricane caused damage”) as emotional tone unless evaluative framing is present.


OBJECTIVE 2 — Contextual Sentiment Stabilization (CSS)
Minimize: Misclassifying factual negativity or criticism as emotional polarity.

Rules:
- Reporting on negative events is not automatically Negative sentiment.
- Quoted opinions must not override the author’s overall tone.
- Balanced coverage with mixed language should default to Neutral.
- Use ONLY language present in the text.
- Never infer author intent beyond explicit wording.
- If polarity is ambiguous or evenly balanced → classify as Neutral.


FRACTAL REASONING PHASES (Internal)

LAYER 1 — LOCAL SIGNAL TAGGING (Micro-Level Computation)
Tag each sentence as: Positive evaluative, Negative evaluative, or Neutral factual based only on explicit evaluative language (praise/blame, optimism/pessimism).

LAYER 2 — LOCAL ERROR MINIMIZATION
Correct for confounds:

Negative events described factually (not evaluative)

Quoted opinions vs author narrative voice

Technical/clinical language that sounds negative but is neutral
Recompute the balance of positive/negative evaluative cues after correction.

LAYER 3 — APERTURE EXPANSION (Document-Level Context Integration)
Assess sentiment over the full document:

Does tone shift (e.g., neutral reporting → editorializing)?

Are evaluative cues concentrated in one section or consistent throughout?
Choose label based on dominant overall tone, not isolated sentences.

LAYER 4 — FRACTAL CONSISTENCY CHECK (Cross-Factor Separation)
Enforce separation rules:

Toxicity ≠ sentiment (hostility doesn’t automatically define polarity).

Sensationalism ≠ sentiment (hype can be neutral/negative/positive).

Political stance does not determine sentiment label.
If sentiment conclusion depended on these, revise using only evaluative tone.

LAYER 5 — INTER-AGENT REFLECTIVE CHECK
Simulate two internal evaluators:

Polarity Counter: “Which side has more explicit evaluative language overall?”

Neutral Editor: “Would a neutral reader call this overall positive/negative, or just factual?”
If disagreement exists, default conservatively to Neutral unless polarity is clear.

LAYER 6 — TEMPORAL RE-GROUNDING
Re-check the ending and any late paragraphs for tone changes (e.g., concluding judgment, call-to-action, moral framing).
If later text changes the dominant tone, revise the final label.

FINAL CONSTRAINT:
Do not rely on topic domain knowledge.
Classify sentiment based only on textual tone.
If the article uses sarcasm, irony, or satirical phrasing, do not automatically classify this as sensationalism. Sarcastic language may include exaggerated wording for rhetorical effect without actually misrepresenting facts. Only treat sarcasm as sensationalism if the sarcastic framing meaningfully distorts the article’s claims, exaggerates the implications of events, or attempts to manipulate reader perception through mockery or hyperbolic insinuation.
"""

def get_sentiment_prompt():
    prompt = (
        "Analyze the sentiment of the text.\n\n"
        "Use the Fractal Chain-of-Thought protocol below:\n"
        f"{SENTIMENT_FCOT}\n\n"
        "STEPS:\n"
        "1. Call the `sentiment_predictive_score` tool to get the statistical model's prediction.\n"
        "2. Perform your own analysis of the text using the RUBRIC below.\n"
        "3. Use the `combine_scores` tool to merge the predictive label and your generative label (use is_numeric=False).\n\n"
        f"RUBRIC:\n{SENTIMENT_RUBRIC}\n\n"
        "Choose exactly one label: Positive, Negative, or Neutral. Return JSON with 'predictive_label', 'generative_label', and 'final_label'."
    )
    if USE_XML_PROMPTS:
        return f"<system_instruction>\n{prompt}\n</system_instruction>"
    return prompt


# --- TOXICITY ---
TOXICITY_RUBRIC = """
Classify the toxicity level of the text.
- Friendly: Respectful, polite, and constructive.
- Neutral: Matter-of-fact, no toxic elements.
- Rude: Mildly disrespectful or dismissive.
- Toxic: Aggressive, insulting, or hostile.
- Super_Toxic: Hate speech, threats, or extreme dehumanization.
"""

TOXICITY_FCOT = """
You are a Hostility Severity Detection Agent implementing dual-objective optimization.

DUAL OBJECTIVE FUNCTIONS

OBJECTIVE 1 — Explicit Hostility Detection (EHD)
Maximize: Accurate detection of explicit hostile language.

Definitions:
Identify whether the text contains:
- Direct insults toward an individual or group
- Profanity used to demean
- Dehumanizing language
- Aggressive hostility
- Threats or hate speech

Severity hierarchy (use highest detected level):

Level 0: No hostility detected.
Level 1: Mild dismissiveness or sarcastic belittling.
Level 2: Direct insult or derogatory language.
Level 3: Aggressive hostility or dehumanization.
Level 4: Explicit threats or hate speech.

Mapping:
Level 0 → Friendly
Level 1 → Neutral
Level 2 → Rude
Level 3 → Toxic
Level 4 → Super_Toxic


OBJECTIVE 2 — Contextual Filtering Constraint (CFC)
Minimize: False positives caused by contextual misinterpretation.

Rules:
- Quoted toxic speech is NOT toxic unless endorsed by the author.
- Reporting on hateful statements is not the same as producing them.
- Strong criticism is not automatically toxic.
- Negative sentiment alone does NOT imply toxicity.
- Use ONLY the explicit wording in the text.
- Never infer hostility beyond what is directly stated.


FRACTAL REASONING PHASES (Internal)

LAYER 1 — LOCAL SIGNAL TAGGING (Micro-Level Computation)
Scan each sentence and tag explicit toxicity cues as one of:

Insult / Derogatory term

Profanity used to demean

Dehumanization

Threat / incitement / hate speech
Also tag whether each cue targets a person/group or is general.

LAYER 2 — LOCAL ERROR MINIMIZATION
Correct for common false positives:

Quoted speech vs author voice

Reporting/description of toxic language vs endorsing it

Sarcasm without explicit insult

Strong criticism that is not abusive
Retain only cues that remain valid after correction.

LAYER 3 — APERTURE EXPANSION (Document-Level Context Integration)
Use broader context to interpret cues:

Is the cue repeated or central to the message, or incidental?

Does surrounding text condemn, distance, or endorse the cue?

Does toxicity intensify across the text or remain isolated?
Update provisional severity if context changes interpretation.

LAYER 4 — FRACTAL CONSISTENCY CHECK (Cross-Factor Propagation Guard)
Enforce:

Negative sentiment ≠ toxicity unless explicit abusive cues exist.

Political framing ≠ toxicity unless explicit abusive cues exist.

Emotional language only increases toxicity if it includes hostile targeting.
If inconsistency exists, downgrade toxicity accordingly.

LAYER 5 — INTER-AGENT REFLECTIVE CHECK
Simulate two internal evaluators:

Strict Literalist: “Do we have explicit hostile wording that meets the label definition?”

Contextual Auditor: “Is this quoted/reporting/condemned rather than authored hostility?”
If disagreement exists, choose the more conservative (lower) toxicity label.

LAYER 6 — TEMPORAL RE-GROUNDING
Re-scan the full text from start to end to catch late clarifications:

Later lines may reveal quotes, attribution, condemnation, or escalation.
Revise the final label if later context changes authorship/endorsement or severity.

FINAL CONSTRAINT:
If no explicit hostile language is detected, classify as Friendly or Neutral.
"""


def get_toxicity_prompt():
    prompt = (
        "Analyze the toxicity of the text.\n\n"
        "Use the Fractal Chain-of-Thought protocol below:\n"
        f"{TOXICITY_FCOT}\n\n"
        "STEPS:\n"
        "1. Call the `toxicity_predictive_score` tool to get the statistical model's prediction.\n"
        "2. Perform your own analysis of the text using the RUBRIC below.\n"
        "3. Use the `combine_scores` tool to merge the predictive label and your generative label (use is_numeric=False).\n\n"
        f"RUBRIC:\n{TOXICITY_RUBRIC}\n\n"
        "Choose exactly one label: Friendly, Neutral, Rude, Toxic, or Super_Toxic. Return JSON with 'predictive_label', 'generative_label', and 'final_label'."
    )
    if USE_XML_PROMPTS:
        return f"<system_instruction>\n{prompt}\n</system_instruction>"
    return prompt


# --- FINAL VERACITY PROMPT ---
FINAL_VERACITY_PROMPT = """
You are an expert misinformation analysis system.

Your task is to compute a FINAL VERACITY SCORE representing the probability that a news article contains misinformation or misleading framing.

You must evaluate the article using the outputs of previously computed factuality factors and a small number of additional meta-audit checks.

You MUST follow the scoring formula provided and compute the final probability step-by-step.

------------------------------------------------

INPUTS

Article Title:
{title}

Article Body:
{body}

Module Outputs:

Clickbait Score (0-100):
{clickbait_score}

Headline–Body Relation Score (0-100):
{headline_body_score}

Sensationalism Score (0-100):
{sensationalism_score}

Sentiment Label:
{sentiment_label}

Toxicity Label:
{toxicity_label}

Political Affiliation Label:
{political_label}

------------------------------------------------

STEP 1 — NORMALIZE FACTOR RISKS

Convert module outputs into misinformation risk scores between 0 and 1.

Clickbait Risk
clickbait_risk = clickbait_score / 100

Headline-Body Mismatch Risk
hbr_risk = 1 - (headline_body_score / 100)

Sensationalism Risk
sensationalism_risk = sensationalism_score / 100

Toxicity Risk Mapping

Friendly = 0.00  
Neutral = 0.10  
Rude = 0.20  
Toxic = 0.70  
Super_Toxic = 1.00  

Set toxicity_risk according to the label.

Sentiment is NOT directly used as a risk factor, but may inform reasoning in the meta-audit.

Political affiliation must NOT directly increase misinformation risk.

------------------------------------------------

STEP 2 — META-AUDIT CHECKS

Perform the following checks using the article content.

Evidence Weakness

Evaluate whether the article relies on:

• anecdotes instead of evidence  
• testimony or quotes as primary evidence  
• vague or unnamed sources  
• claims without supporting data or explanation  

Assign:

0.0 = strong evidence with multiple sources or data  
0.5 = mixed evidence quality  
1.0 = weak or unsupported claims

Logical Fallacy Detection

Look for reasoning patterns such as:

• strawman arguments/ad hominem attacks: argument attacks distorted version of opponent's stance or the person rather than the claim
• false dichotomy / false equivalence: complex situations oversimplified into binary scenarios to manipulate reader

Echo Chamber / Source Diversity

Check whether the article:

• cites only ideologically aligned voices  
• lacks opposing viewpoints or context  
• repeats claims without verification

META AUDIT SCORE

Combine the above observations into:

meta_audit_risk ∈ [0,1]

Where:

0.0 = no manipulation signals  
0.5 = moderate rhetorical manipulation  
1.0 = strong manipulation or fallacious reasoning

------------------------------------------------

STEP 3 — COMPUTE FINAL MISINFORMATION RISK

Use the following weighted equation:

Final Risk =

0.20 * hbr_risk  
+ 0.15 * evidence_weakness  
+ 0.20 * sensationalism_risk  
+ 0.15 * clickbait_risk  
+ 0.20 * toxicity_risk  
+ 0.10 * meta_audit_risk

------------------------------------------------

STEP 4 — ASSIGN VERACITY LABEL

Interpret the final risk score:

0.00 – 0.29 → Likely Reliable  
0.30 – 0.49 → Somewhat Questionable  
0.50 – 0.69 → Potential Misinformation  
0.70 – 1.00 → High Risk of Misinformation

------------------------------------------------

STEP 5 — OUTPUT STRUCTURED JSON

Return ONLY valid JSON in this format:

{
  "normalized_scores": {
    "clickbait_risk": float,
    "hbr_risk": float,
    "sensationalism_risk": float,
    "toxicity_risk": float
  },
  "meta_audit": {
    "evidence_weakness": float,
    "meta_audit_risk": float,
    "detected_fallacies": [string],
    "echo_chamber_signals": [string]
  },
  "final_score": float,
  "veracity_label": string,
  "reasoning_summary": "2-3 sentence explanation of why the article received this score."
}

Do not output anything except the JSON.
"""

# --- ORCHESTRATOR ---
ORCHESTRATOR_PROMPT = """

    You are the Factuality Orchestrator Agent.
    Your job is to coordinate the specialized factuality-factor agents and then call the Final Veracity Scoring Agent.

    INPUT:
    You will receive an article with a headline and body.

    REQUIRED PROCESS:
    1. Call the clickbait agent on the headline.
    2. Call the headline-body agent on the headline and body.
    3. Call the political affiliation agent on the full article text.
    4. Call the sensationalism agent on the full article text.
    5. Call the sentiment agent on the full article text.
    6. Call the toxicity agent on the full article text.
    7. Collect their final outputs:
    - clickbait final_score
    - headline_body final_score
    - political final_label
    - sensationalism final_score
    - sentiment final_label
    - toxicity final_label
    8. Call the final_veracity_agent using:
    - title = headline
    - body = body
    - clickbait_score = clickbait final_score
    - headline_body_score = headline_body final_score
    - sensationalism_score = sensationalism final_score
    - sentiment_label = sentiment final_label
    - toxicity_label = toxicity final_label
    - political_label = political final_label
    9. Return one final JSON object that includes:
    - all module outputs
    - the final veracity output

    OUTPUT REQUIREMENTS:
    - Return ONLY valid JSON.
    - Do not include markdown.
    - Do not omit any sub-agent outputs.
    """

