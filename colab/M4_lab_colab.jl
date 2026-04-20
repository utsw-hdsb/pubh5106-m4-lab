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
#     display_name: Julia
#     language: julia
#     name: julia
# ---

# %% [markdown]
# # M4 Lab: The Diagnostician
#
# **PUBH 5106 — AI in Health Applications**
#
# **Platform:** Google Colab (Julia runtime)
#
# ## Colab Setup Instructions
#
# 1. In the menu bar, go to **Runtime → Change runtime type**
# 2. Under **Runtime type**, select **Julia**
# 3. Click **Save**
# 4. Run the setup cell below — it installs packages and downloads
#    lab files (~2-3 minutes the first time)

# %% [markdown]
# ---
# ## Setup (Run This First)

# %%
# === COLAB SETUP — installs packages and downloads lab files ===
# This cell takes 2-3 minutes on first run.

using Pkg

# Install required packages
println("Installing packages...")
Pkg.add(["BayesNets", "HTTP", "JSON3", "DataFrames"])
println("Packages installed ✓")

# Download lab files from GitHub
println("\nDownloading lab files...")
REPO_URL = "https://raw.githubusercontent.com/utsw-hdsb/pubh5106-m4-lab/main"

mkpath("data")

for f in ["lab_utils.jl",
          "data/vignettes.json",
          "data/llm_precomputed.json",
          "data/asia_network.svg",
          "data/child_network.svg"]
    url = "$(REPO_URL)/$(f)"
    try
        download(url, f)
        println("  $(f) ✓")
    catch e
        println("  $(f) FAILED: $(e)")
    end
end

println("\nSetup complete!")

# %%
include("lab_utils.jl")
using .LabUtils
using BayesNets
using JSON3

# Bring all exported names into scope explicitly
# (Colab's cell-by-cell execution can lose module scope)
import .LabUtils: set_api_keys, call_llm, ask_llm_probability,
    load_precomputed_llm, brier_score, calibration_error,
    score_round, show_calibration, submit_to_leaderboard,
    build_asia_network, build_child_network,
    query_bn, query_bn_all, bn_node_names, bn_parents,
    verify_setup, BNCategorical

# %% [markdown]
# ### Your Group Name and API Keys
#
# Set your group name and register your team's Groq API keys.
# Each team member should create a free account at
# [console.groq.com](https://console.groq.com) and generate an API key.

# %%
LabUtils.GROUP_NAME[] = "CHANGE_ME"  # <-- Set your group name here

# %%
set_api_keys([
    "gsk_...",  # Team member 1
    "gsk_...",  # Team member 2
    "gsk_...",  # Team member 3
])

# %% [markdown]
# ### Verify Setup

# %%
LabUtils.DATA_DIR[] = "data"
verify_setup()

# %% [markdown]
# ---
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
#
# ## How This Lab Works
#
# This is a **competitive group lab** scored on **calibration** — how well
# your probability estimates match reality.
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
# *The remainder of this notebook is identical to the Codespaces version.
# All cells below this point work the same way.*
#
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
# ground truth using the Brier score.

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
r1_result = score_round(r1_human, [round(Int, t) for t in r1_truth .>= 0.5])
println("Your Brier score: $(brier_score(r1_human, [round(Int, t) for t in r1_truth .>= 0.5]))")

println("\n  Case  Your Est.  Truth     Error")
println("  " * "-"^40)
for (i, case) in enumerate(r1_cases)
    err = abs(r1_human[i] - case.ground_truth)
    println("  $(i)     $(r1_human[i])       $(case.ground_truth)     $(round(err; digits=2))")
end

println("\n--- Explanations ---")
for (i, case) in enumerate(r1_cases)
    println("  Case $(i): $(case.explanation)")
end

submit_to_leaderboard(1, r1_result)

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
# The network has 8 nodes and 18 parameters.
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

# Root nodes — done for you as an example
push!(asia, DiscreteCPD(:Smoker, [0.5, 0.5]))
push!(asia, DiscreteCPD(:Asia, [0.99, 0.01]))

# TODO: Add LungCancer node (parent: Smoker)
# Hint: Use CategoricalCPD(:LungCancer, [:Smoker], [2], [...])
# State 1 = no, State 2 = yes. Provide a BNCategorical for each parent state.


# TODO: Add Bronchitis node (parent: Smoker)


# TODO: Add Tuberculosis node (parent: Asia)


# TODO: Add TbOrCancer node (parents: Tuberculosis, LungCancer)
# This is a deterministic OR gate: P(TbOrCancer=yes) = 1.0 if either
# parent is yes, 0.0 only if both parents are no.


# TODO: Add XRay node (parent: TbOrCancer)


# TODO: Add Dyspnoea node (parents: TbOrCancer, Bronchitis)


println("ASIA network: $(length(bn_node_names(asia))) nodes")

# %% [markdown]
# ### 2.2 Visualize Your Network

# %%
using Markdown
display(Markdown.parse("![ASIA Network](data/asia_network.svg)"))

# %% [markdown]
# ### 2.3 Query the Network

# %%
# Prior probabilities (no evidence)
println("=== Prior Probabilities (no evidence) ===")
for node in [:LungCancer, :Tuberculosis, :Bronchitis]
    p = query_bn(asia, node)
    println("  P($(node) = yes) = $(round(p[2]; digits=4))")
end

# %%
# What if the patient is a smoker with an abnormal X-ray?
println("=== Smoker with abnormal X-ray ===")
evidence = Assignment(:Smoker => 2, :XRay => 2)
for node in [:LungCancer, :Tuberculosis, :Bronchitis]
    p = query_bn(asia, node; evidence=evidence)
    println("  P($(node) = yes | Smoker, abnormal XRay) = $(round(p[2]; digits=4))")
end

# %%
# What if they also visited Asia?
println("=== Smoker + abnormal X-ray + visited Asia ===")
evidence2 = Assignment(:Smoker => 2, :XRay => 2, :Asia => 2)
for node in [:LungCancer, :Tuberculosis, :Bronchitis]
    p = query_bn(asia, node; evidence=evidence2)
    println("  P($(node) = yes | Smoker, XRay, Asia) = $(round(p[2]; digits=4))")
end

# %% [markdown]
# **Discussion:** Notice how adding the Asia travel history shifts
# probability toward TB and *away* from lung cancer — even though the
# X-ray finding is the same. This is Bayesian reasoning in action.

# %%
r2_result = Dict("composite" => length(bn_node_names(asia)) == 8 ? 100.0 : 0.0,
                 "brier_score" => 0.0, "calibration_error" => 0.0,
                 "n_cases" => 1)
submit_to_leaderboard(2, r2_result)

# %% [markdown]
# ---
# ## Round 3: Head to Head (~35 min)
#
# **The CHILD network** (20 nodes, 230+ parameters) — a pre-built
# Bayesian network for diagnosing congenital heart disease in newborns
# (Spiegelhalter & Cowell, 1992).
#
# You will compare three sources on the same clinical cases:
# 1. **Your intuition**
# 2. **The Bayesian network**
# 3. **An LLM** (via Groq API)

# %%
child = build_child_network()
println("CHILD network: $(length(bn_node_names(child))) nodes")
display(Markdown.parse("![CHILD Network](data/child_network.svg)"))

# %%
child_cases = vignettes.child_cases

println("=== Round 3: Head to Head ===\n")
for case in child_cases
    println("$(case.label): $(case.description)\n")
end

# %% [markdown]
# ### 3.1 Your Estimates (0.0 to 1.0)

# %%
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
r3_truth = [1, 1, 1]
r3_bn_estimates = Float64.(r3_bn_results)
r3_llm_estimates = Float64.(r3_llm_results)

show_calibration("Round 3 — Head to Head",
                 r3_bn_estimates, r3_llm_estimates, r3_human,
                 r3_truth)

# %%
r3_result = score_round(r3_bn_estimates, r3_truth)
submit_to_leaderboard(3, r3_result)

# %% [markdown]
# ### 3.5 Stress Test: Base Rate Shift

# %%
stress = vignettes.stress_test

println("=== Base Rate Shift ===")
println(stress.base_rate_shift.description)

orig_ev = Assignment(Dict(Symbol(k) => v
    for (k, v) in pairs(stress.base_rate_shift.original_evidence)))
p_orig = query_bn(child, :Disease; evidence=orig_ev)

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

min_ev = Assignment(Dict(Symbol(k) => v
    for (k, v) in pairs(stress.missing_evidence.evidence)))
p_minimal = query_bn(child, :Disease; evidence=min_ev)

p_prior = query_bn(child, :Disease)

println("\n  Disease       Prior      Grunting Only")
println("  " * "-"^45)
for (i, d) in enumerate(diseases)
    println("  $(rpad(d, 12))  $(round(p_prior[i]; digits=3))      $(round(p_minimal[i]; digits=3))")
end

# %%
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

# %% [markdown]
# ---
# ## Reflection
#
# ### The Three Paradigms Compared
#
# | Paradigm | Tool | Strength | Weakness |
# |----------|------|----------|----------|
# | **ML (LLMs)** | llama3.1 | Flexible, fluent | Overconfident, no calibrated uncertainty |
# | **Logic** | KGs, rules | Auditable, traceable | Brittle, no uncertainty |
# | **Probability** | BNs | Calibrated uncertainty | Requires structure + data |
#
# The most trustworthy clinical AI systems draw on **all three**.

# %% [markdown]
# **R1: When Should AI Say "I Don't Know"?**

# %%
# YOUR RESPONSE:


# %% [markdown]
# **R2: Could You Trust This Network?**
#
# The CHILD network was published in 1992. Would you deploy it in a NICU
# today without changes? What would you need to update?

# %%
# YOUR RESPONSE:


# %% [markdown]
# **R3: The Three-Paradigm Synthesis**
#
# How would you combine a knowledge graph, a Bayesian network, and an LLM
# for a clinical decision? What does each contribute?

# %%
# YOUR RESPONSE:


# %% [markdown]
# ---
# ## Submission
#
# Download this notebook (File → Download → Download .ipynb) and email it
# to **brian.chapman@utsouthwestern.edu**.
