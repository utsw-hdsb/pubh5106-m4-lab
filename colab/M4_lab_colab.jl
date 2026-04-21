# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: jl:percent,ipynb
#     text_representation:
#       extension: .jl
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Julia 1.12
#     language: julia
#     name: julia-1.12
# ---

# %% [markdown]
# # M4 Lab: The Diagnostician
#
# **PUBH 5106 — AI in Health Applications**
#
# **Platform:** GitHub Codespaces
#
# ## Learning Objectives
#
# By the end of this lab, you will be able to:
#
# 1. Apply Bayes' theorem to estimate disease probability from clinical
#    findings and recognize the base rate neglect bias
# 2. Build a Bayesian network from a published clinical model, specify
#    conditional probability tables, and query it with patient evidence
# 3. Compare the calibrated uncertainty of a Bayesian network to the
#    stated confidence of a large language model on the same clinical cases
# 4. Explain why Bayesian networks respond appropriately to base rate
#    shifts and missing evidence while LLMs do not
# 5. Evaluate the strengths and limitations of each reasoning approach
#    (human intuition, Bayesian network, LLM) for clinical decision support
#
# ## How This Lab Works
#
# This is a **competitive group lab** scored on **calibration** — how well
# your probability estimates match reality. The best team is not the most
# confident, but the most *accurately* confident.
#
# You will use three tools side by side:
# - **Your intuition** — what you think the probability is
# - **A Bayesian network** — what the math says
# - **An LLM** — what a language model says
#
# | Round | Name | Time | What You Do |
# |-------|------|------|-------------|
# | — | Warm-Up | ~10 min | Explore the Aronsky & Haug pneumonia network |
# | 1 | Base Rate Trap | ~15 min | Estimate probabilities by intuition |
# | 2 | Build the Network | ~25 min | Construct the ASIA Bayesian network |
# | 3 | Head to Head | ~35 min | BN vs. LLM on neonatal cases + stress test |
# | — | Reflection | ~15 min | Leaderboard + discussion |
#
# **Scoring metric:** Brier score — lower is better. Measures how close
# your stated probability is to the ground truth. A perfect predictor
# scores 0; random guessing scores 0.25.

# %% [markdown]
# ---
# ## Setup

# %%
using Pkg
Pkg.activate(".")

include("lab_utils.jl")
using .LabUtils
using BayesNets

using JSON3

# %% [markdown]
# ### Your Group Name

# %%
LabUtils.GROUP_NAME[] = "CHANGE_ME"  # <-- Set your group name here

# %% [markdown]
# ### Verify Setup

# %%
LabUtils.DATA_DIR[] = "data"
verify_setup()

# %% [markdown]
# ---
# ## Warm-Up: A Production-Scale Clinical Bayesian Network (~10 min)
#
# Before building anything, look at what a real clinical Bayesian network
# looks like.
#
# **Aronsky & Haug (1998)** built a Bayesian network for pneumonia
# diagnosis at LDS Hospital in Salt Lake City. It was trained on data
# from 32,000+ emergency department patients and achieved:
# - Sensitivity: 95%
# - Specificity: 96.5%
# - AUC: 0.98
#
# The network has 25 nodes across 7 categories of clinical data (blood
# gas, demographics, history, radiology, vitals, lab values, nurse
# assessment) and 10,100 conditional probability parameters — all learned
# from patient data.
#
# **Discussion questions:**
# 1. What kinds of clinical knowledge are encoded in the network structure
#    (the arrows)?
# 2. Could you build this network without domain expertise? What would a
#    purely data-driven approach miss?
# 3. The CPTs were learned from LDS Hospital patients in 1995–1997. Would
#    this network work at UTSW today? What might have changed?

# %% [markdown]
# ---
# ## Round 1: Base Rate Trap (~15 min)
#
# **No tools.** Just your clinical intuition.
#
# You will read 5 clinical vignettes and estimate the probability that
# each patient has pneumonia. Write a number between 0.0 and 1.0.
#
# **Scoring:** Your estimates will be scored against epidemiological
# ground truth using the Brier score. The point is not to get them
# "right" — it's to see where your intuition is well-calibrated and
# where it's biased.
#
# The most common bias you'll encounter: **base rate neglect** — the
# tendency to focus on symptoms and ignore how common the disease is
# in the clinical setting.

# %%
# Load vignettes
vignettes = JSON3.read(read(joinpath(LabUtils.DATA_DIR[], "vignettes.json"), String))
r1_cases = vignettes.round1_base_rate

println("=== Round 1: Base Rate Trap ===\n")
for (i, case) in enumerate(r1_cases)
    println("Case $(i): $(case.vignette)")
    println("  Question: $(case.question)\n")
end

# %% [markdown]
# ### 1.1 Your Estimates
#
# Fill in your probability estimates below (0.0 to 1.0).

# %%
r1_human = [0.0, 0.0, 0.0, 0.0, 0.0]  # <-- Fill in your estimates

# Ground truth
r1_truth = [case.ground_truth for case in r1_cases]

# %% [markdown]
# ### 1.2 Score Your Intuition

# %%
r1_result = score_round(r1_human, [Int(t >= 0.5) for t in r1_truth])
println("Your Brier score: $(brier_score(r1_human, [Int(t >= 0.5) for t in r1_truth]))")

# Show per-case comparison
println("\n  Case  Your Est.  Truth     Error")
println("  " * "-"^40)
for (i, case) in enumerate(r1_cases)
    err = abs(r1_human[i] - case.ground_truth)
    println("  $(i)     $(r1_human[i])       $(case.ground_truth)     $(round(err; digits=2))")
end

# Show explanations
println("\n--- Explanations ---")
for (i, case) in enumerate(r1_cases)
    println("  Case $(i): $(case.explanation)")
end

submit_to_leaderboard(1, r1_result)

# %% [markdown]
# **Discussion:** Where were you most wrong? The base rate trap is most
# visible in Cases 1 vs. 2 — the *same symptom* (cough) has very different
# implications depending on the clinical setting (ED vs. outpatient).
# Bayes' theorem formalizes this: the prior probability (base rate) is
# just as important as the likelihood (symptoms).

# %% [markdown]
# ---
# ## Round 2: Build the Network (~25 min)
#
# **Build the ASIA Bayesian network** from Lauritzen & Spiegelhalter (1988).
#
# This classic teaching network models the differential diagnosis of
# shortness of breath (dyspnoea). The possible causes are:
# - **Tuberculosis** (increased risk if recent visit to Asia)
# - **Lung cancer** (increased risk if smoker)
# - **Bronchitis** (increased risk if smoker)
#
# The network has 8 nodes and 18 parameters. You will build it from
# scratch using BayesNets.jl.
#
# ### The Network Structure
#
# ```
#   Asia          Smoker
#    |           /      \
#    v          v        v
#   TB    LungCancer   Bronchitis
#     \      /              |
#      v    v               |
#    TbOrCancer             |
#      /      \             |
#     v        v            v
#   XRay      Dyspnoea <----
# ```
#
# ### The Probabilities
#
# | Node | Parents | CPT |
# |------|---------|-----|
# | Asia | — | P(visited) = 0.01 |
# | Smoker | — | P(smoker) = 0.50 |
# | Tuberculosis | Asia | P(TB\|Asia=yes) = 0.05, P(TB\|Asia=no) = 0.01 |
# | LungCancer | Smoker | P(LC\|Smoke=yes) = 0.10, P(LC\|Smoke=no) = 0.01 |
# | Bronchitis | Smoker | P(Br\|Smoke=yes) = 0.60, P(Br\|Smoke=no) = 0.30 |
# | TbOrCancer | TB, LC | Deterministic OR: 1 if either TB or LC is present |
# | XRay | TbOrCancer | P(abnormal\|TbOrCa=yes) = 0.98, P(abnormal\|TbOrCa=no) = 0.05 |
# | Dyspnoea | TbOrCancer, Bronchitis | See table below |
#
# **Dyspnoea CPT:**
#
# | TbOrCancer | Bronchitis | P(Dyspnoea = yes) |
# |------------|------------|-------------------|
# | yes | yes | 0.90 |
# | yes | no | 0.70 |
# | no | yes | 0.80 |
# | no | no | 0.10 |
#
# **Encoding:** State 1 = no/absent, State 2 = yes/present.

# %% [markdown]
# ### 2.1 Build the Network

# %%
asia = DiscreteBayesNet()

# Root nodes (no parents) — these are done for you as an example
push!(asia, DiscreteCPD(:Smoker, [0.5, 0.5]))
push!(asia, DiscreteCPD(:Asia, [0.99, 0.01]))  # P(no)=0.99, P(yes)=0.01

# TODO: Add LungCancer node (parent: Smoker)
# Hint: Use CategoricalCPD(:LungCancer, [:Smoker], [2], [...])
# State 1 = no, State 2 = yes. Provide a BNCategorical for each parent state.


# TODO: Add Bronchitis node (parent: Smoker)


# TODO: Add Tuberculosis node (parent: Asia)


# TODO: Add TbOrCancer node (parents: Tuberculosis, LungCancer)
# This is a deterministic OR gate: P(TbOrCancer=yes) = 1.0 if either
# parent is yes, 0.0 only if both parents are no.
# Parent state ordering: (TB=yes,LC=yes), (TB=yes,LC=no),
#                        (TB=no,LC=yes), (TB=no,LC=no)


# TODO: Add XRay node (parent: TbOrCancer)


# TODO: Add Dyspnoea node (parents: TbOrCancer, Bronchitis)
# Use the Dyspnoea CPT table from the markdown above.


println("ASIA network: $(length(bn_node_names(asia))) nodes")

# %% [markdown]
# ### 2.2 Visualize Your Network

# %%
# Display the ASIA network structure
using Markdown
display(Markdown.parse("![ASIA Network](data/asia_network.svg)"))

# %% [markdown]
# ### 2.3 Query the Network
#
# Test your network with some clinical scenarios.

# %%
# Check your network is complete before querying
if length(bn_node_names(asia)) < 8
    println("⚠ Your network has $(length(bn_node_names(asia)))/8 nodes.")
    println("  Complete the TODO cells above before running queries.")
    println("  Missing: $(setdiff([:Asia,:Smoker,:LungCancer,:Bronchitis,:Tuberculosis,:TbOrCancer,:XRay,:Dyspnoea], bn_node_names(asia)))")
else
    println("✓ Network complete: $(length(bn_node_names(asia))) nodes")
end

# %%
# Prior probabilities (no evidence)
println("=== Prior Probabilities (no evidence) ===")
for node in [:LungCancer, :Tuberculosis, :Bronchitis]
    if node in bn_node_names(asia)
        p = query_bn(asia, node)
        println("  P($(node) = yes) = $(round(p[2]; digits=4))")
    else
        println("  $(node): not yet added to network")
    end
end

# %%
# What if the patient is a smoker with an abnormal X-ray?
# State 2 = yes/present for all nodes
if length(bn_node_names(asia)) == 8
    println("=== Smoker with abnormal X-ray ===")
    evidence = Assignment(:Smoker => 2, :XRay => 2)
    for node in [:LungCancer, :Tuberculosis, :Bronchitis]
        p = query_bn(asia, node; evidence=evidence)
        println("  P($(node) = yes | Smoker, abnormal XRay) = $(round(p[2]; digits=4))")
    end
else
    println("⚠ Complete the network (8 nodes) before running evidence queries.")
end

# %%
# What if they also visited Asia?
if length(bn_node_names(asia)) == 8
    println("=== Smoker + abnormal X-ray + visited Asia ===")
    evidence2 = Assignment(:Smoker => 2, :XRay => 2, :Asia => 2)
    for node in [:LungCancer, :Tuberculosis, :Bronchitis]
        p = query_bn(asia, node; evidence=evidence2)
        println("  P($(node) = yes | Smoker, XRay, Asia) = $(round(p[2]; digits=4))")
    end
else
    println("⚠ Complete the network (8 nodes) before running evidence queries.")
end

# %% [markdown]
# **Discussion:** Notice how adding the Asia travel history shifts
# probability toward TB and *away* from lung cancer — even though the
# X-ray finding is the same. This is Bayesian reasoning in action: new
# evidence updates all related probabilities simultaneously through the
# network structure.
#
# The **TbOrCancer** node is a deterministic OR gate — it is exactly 1
# (true) if either TB or LungCancer is present. This is a logical
# operation embedded in a probabilistic network. It bridges Module 3
# (logic) and Module 4 (probability): the same system can encode both
# certain logical rules and uncertain probabilistic relationships.

# %%
# Submit Round 2 (network construction is pass/fail)
r2_result = Dict("composite" => length(bn_node_names(asia)) == 8 ? 100.0 : 0.0,
                 "brier_score" => 0.0, "calibration_error" => 0.0,
                 "n_cases" => 1)
submit_to_leaderboard(2, r2_result)

# %% [markdown]
# ---
# ## Round 3: Head to Head (~25 min)
#
# **Unlocked:** The CHILD network (20 nodes, 230 parameters) — a
# pre-built Bayesian network for diagnosing congenital heart disease
# in newborns (Spiegelhalter & Cowell, 1992).
#
# You will present the same patient cases to three sources:
# 1. **Your intuition** — estimate the most likely diagnosis
# 2. **The Bayesian network** — query the CHILD network
# 3. **An LLM** — ask the language model
#
# Then compare how well-calibrated each source is.
#
# ### The CHILD Network
#
# The network models 6 possible diagnoses:
# 1. PFC (persistent fetal circulation)
# 2. TGA (transposition of great arteries)
# 3. Fallot (tetralogy of Fallot)
# 4. PAIVS (pulmonary atresia with intact ventricular septum)
# 5. TAPVD (total anomalous pulmonary venous drainage)
# 6. Lung disease

# %%
# Load the CHILD network (pre-built, no file parsing needed)
child = build_child_network()
println("CHILD network: $(length(bn_node_names(child))) nodes")

# Display the network structure
display(Markdown.parse("![CHILD Network](data/child_network.svg)"))

# %%
# Load the clinical cases
child_cases = vignettes.child_cases

println("=== Round 3: Head to Head ===\n")
for case in child_cases
    println("$(case.label): $(case.description)\n")
end

# %% [markdown]
# ### 3.1 Your Estimates
#
# For each case, which of the 6 diagnoses is most likely? Estimate the
# probability of the most likely diagnosis (0.0 to 1.0).

# %%
# TODO: Fill in your probability estimates for each case (0.0 to 1.0)
# One estimate per case — your confidence in the most likely diagnosis.
r3_human = [0.0, 0.0, 0.0]  # <-- Fill in your estimates

# %% [markdown]
# ### 3.2 Query the Bayesian Network

# %%
r3_bn_results = []
for case in child_cases
    evidence = Assignment(Dict(Symbol(k) => v for (k, v) in pairs(case.evidence)))
    p_disease = query_bn(child, Symbol(case.target); evidence=evidence)

    println("$(case.label): P(Disease) = $(round.(p_disease; digits=3))")
    diseases = ["PFC", "TGA", "Fallot", "PAIVS", "TAPVD", "Lung"]
    best_idx = argmax(p_disease)
    println("  Most likely: $(diseases[best_idx]) ($(round(p_disease[best_idx]; digits=3)))")
    println("  Ground truth: $(case.ground_truth_disease)\n")
    push!(r3_bn_results, p_disease[case.ground_truth_idx])
end

# %% [markdown]
# ### 3.3 Ask the LLM

# %%
r3_llm_results = Float64[]
for case in child_cases
    p = ask_llm_probability(case.description, case.ground_truth_disease)
    println("$(case.label): LLM P($(case.ground_truth_disease)) = $(p)")
    push!(r3_llm_results, p)
end

# %% [markdown]
# ### 3.4 Compare Calibration

# %%
# Ground truth: was the BN's top diagnosis correct?
r3_truth = [1, 1, 1]  # All cases have known diagnoses

r3_bn_estimates = Float64.(r3_bn_results)
r3_llm_estimates = Float64.(r3_llm_results)

show_calibration("Round 3 — Head to Head",
                 r3_bn_estimates, r3_llm_estimates, r3_human,
                 r3_truth)

# %%
r3_result = score_round(r3_bn_estimates, r3_truth)
submit_to_leaderboard(3, r3_result)

# %% [markdown]
# **Discussion:**
# - Which source was most confident? Most accurate?
# - Did the LLM express appropriate uncertainty, or was it overconfident?
# - The BN's probabilities come from the CPTs — they are *derived from
#   data*. The LLM's probabilities come from... where?

# %% [markdown]
# ### 3.5 Stress Test: Base Rate Shift
#
# Same patient evidence, but now change the prior — what if this newborn
# had birth asphyxia? The BN's prior P(Disease | BirthAsphyxia) shifts
# the entire posterior. Does the LLM adjust?

# %%
stress = vignettes.stress_test

println("=== Base Rate Shift ===")
println(stress.base_rate_shift.description)

# Query with original evidence (no birth asphyxia)
orig_ev = Assignment(Dict(Symbol(k) => v
    for (k, v) in pairs(stress.base_rate_shift.original_evidence)))
p_orig = query_bn(child, :Disease; evidence=orig_ev)

# Query with shifted evidence (birth asphyxia = yes)
shift_ev = Assignment(Dict(Symbol(k) => v
    for (k, v) in pairs(stress.base_rate_shift.shifted_evidence)))
p_shift = query_bn(child, :Disease; evidence=shift_ev)

diseases = ["PFC", "TGA", "Fallot", "PAIVS", "TAPVD", "Lung"]
println("\n  Disease       Original   Shifted    Change")
println("  " * "-"^50)
for (i, d) in enumerate(diseases)
    diff = p_shift[i] - p_orig[i]
    println("  $(rpad(d, 12))  $(round(p_orig[i]; digits=3))      $(round(p_shift[i]; digits=3))      $(round(diff; sigdigits=2))")
end

# %%
# Ask the LLM both versions
println("\n=== LLM Response ===")
println("Original (no asphyxia):")
llm_orig = ask_llm_probability(
    "2-day-old newborn, no birth asphyxia. Grunting observed. Chest X-ray shows plethoric lung fields.",
    "TGA")
println("Shifted (with asphyxia):")
llm_shift = ask_llm_probability(
    "2-day-old newborn WITH birth asphyxia. Grunting observed. Chest X-ray shows plethoric lung fields.",
    "TGA")
println("\nBN shift: $(round(p_shift[2] - p_orig[2]; sigdigits=3))")
println("LLM shift: $(round(llm_shift - llm_orig; sigdigits=3))")

# %% [markdown]
# ### 3.6 Stress Test: Missing Evidence

# %%
println("=== Missing Evidence ===")
println(stress.missing_evidence.description)

# BN with minimal evidence
min_ev = Assignment(Dict(Symbol(k) => v
    for (k, v) in pairs(stress.missing_evidence.evidence)))
p_minimal = query_bn(child, :Disease; evidence=min_ev)

# Compare to prior (no evidence at all)
p_prior = query_bn(child, :Disease)

println("\n  Disease       Prior      Grunting Only")
println("  " * "-"^45)
for (i, d) in enumerate(diseases)
    println("  $(rpad(d, 12))  $(round(p_prior[i]; digits=3))      $(round(p_minimal[i]; digits=3))")
end

# %%
# LLM with minimal evidence
println("\nLLM with minimal evidence:")
llm_minimal = ask_llm_probability(
    "A newborn presents with grunting only. No X-ray available, no O2 measurements, birth asphyxia status unknown.",
    "TGA")
println("BN P(TGA | grunting only) = $(round(p_minimal[2]; digits=3))")
println("LLM P(TGA | grunting only) = $(llm_minimal)")

# %% [markdown]
# **Discussion:**
# - The BN's posterior with minimal evidence stays close to the prior —
#   it *knows it doesn't have enough information*. Did the LLM do the same?
# - The base rate shift changed the BN's entire posterior distribution.
#   The LLM likely gave similar answers regardless. Why?
# - This is the core M4 message: **calibrated uncertainty matters more
#   than confident answers** in clinical decision support.

# %% [markdown]
# ---
# ## Final Leaderboard and Reflection (~15 min)
#
# Your instructor will display the final leaderboard.

# %% [markdown]
# ### The Three Paradigms Compared
#
# Over four modules, you have used all three AI paradigms from Module 1:
#
# | Paradigm | Module | Tool | Strength | Weakness |
# |----------|--------|------|----------|----------|
# | **ML (LLMs)** | M1, M2 | llama3.2, qwen2.5 | Flexible, fluent, handles unstructured text | Overconfident, no calibrated uncertainty, hallucinates |
# | **Logic** | M3 | System prompts, Gilda, KGs | Explicit, auditable, traceable | Brittle, requires complete knowledge, no uncertainty |
# | **Probability** | M4 | Bayesian networks | Calibrated uncertainty, responds to evidence, principled | Requires known structure, CPTs from data, computationally expensive at scale |
#
# The most trustworthy clinical AI systems draw on **all three**:
# - Logic to encode clinical rules and constraints
# - Probability to quantify uncertainty
# - ML to handle unstructured data and scale

# %% [markdown]
# ### Reflection Questions
#
# Discuss in your group and write your responses below.

# %% [markdown]
# **R1: When Should AI Say "I Don't Know"?**
#
# In the stress test, the BN responded to missing evidence by widening
# its uncertainty (staying close to the prior). The LLM likely remained
# confident. For a clinical decision support system, which behavior
# is safer? When is uncertainty more valuable than a confident answer?

# %%
# YOUR RESPONSE:
#

# %% [markdown]
# **R2: Could You Trust This Network?**
#
# The CHILD network was published in 1992. Its CPTs were estimated from
# clinical data available at that time. Would you deploy it in a NICU
# today without changes? What would you need to update? How does this
# compare to retraining an LLM on newer data?

# %%
# YOUR RESPONSE:
#

# %% [markdown]
# **R3: The Three-Paradigm Synthesis**
#
# A patient presents with ambiguous symptoms. You have three tools:
# - A knowledge graph (M3) that says: "Drug X treats Disease Y"
# - A Bayesian network (M4) that says: "P(Disease Y | symptoms) = 0.35"
# - An LLM (M2) that says: "This patient likely has Disease Y"
#
# How would you combine these three sources for a clinical decision?
# What does each one contribute that the others cannot?

# %%
# YOUR RESPONSE:
#

# %% [markdown]
# ---
# ## Submission
#
# Save this notebook and email it to **brian.chapman@utsouthwestern.edu**.
# Ensure:
# - All cells have been run
# - Your group name is set correctly
# - Written reflections (R1–R3) are complete
# - All 3 rounds have been scored
