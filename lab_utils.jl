"""
M4 Lab Utilities: The Diagnostician

Infrastructure code for the M4 Bayesian reasoning lab. This module provides:
- Bayesian network construction and inference helpers (BayesNets.jl)
- LLM interaction via Groq API (with key rotation)
- Pre-computed LLM fallback for offline use
- Calibration scoring (Brier score)
- Leaderboard submission
- Visualization helpers (GraphMakie + CairoMakie)

Students import this module and focus on building networks, querying
posteriors, and comparing BN uncertainty to LLM confidence.
"""
module LabUtils

using BayesNets
using HTTP
using JSON3
using DataFrames
using Printf

const BNCategorical = BayesNets.Categorical

export
    # Configuration
    GROUP_NAME, DATA_DIR,
    # API keys
    set_api_keys,
    # LLM
    call_llm, ask_llm_probability,
    # Pre-computed fallback
    load_precomputed_llm,
    # Scoring
    brier_score, calibration_error, score_round, show_calibration,
    # Leaderboard
    submit_to_leaderboard,
    # BN helpers
    BNCategorical,
    build_asia_network, build_child_network,
    query_bn, query_bn_all,
    bn_node_names, bn_parents,
    # Verification
    verify_setup

# ─────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────

const GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
const SMALL_MODEL = "llama-3.1-8b-instant"

GROUP_NAME = Ref("CHANGE_ME")
DATA_DIR = Ref("data")

# Leaderboard (Google Forms) — instructor configures before class
FORM_URL = Ref("https://docs.google.com/forms/d/e/1FAIpQLSdye6IlNCalUqxeXorKXchIEYda7Lv1Q_nZoM3myeHsRPNv_Q/formResponse")
FIELD_GROUP = Ref("entry.580574446")
FIELD_ROUND = Ref("entry.2102634743")
FIELD_SCORE = Ref("entry.1233314305")
FIELD_DETAIL = Ref("entry.1762914451")

# ─────────────────────────────────────────────────────────────────────
# API Key Management
# ─────────────────────────────────────────────────────────────────────

const _api_keys = Ref(String[])
const _current_key_idx = Ref(1)

"""
    set_api_keys(keys::Vector{String})

Register Groq API keys for the session. The system rotates to the
next key on rate-limit (429) errors.

Each team member should create a free account at console.groq.com
and generate an API key.
"""
function set_api_keys(keys::Vector{String})
    _api_keys[] = filter(!isempty, strip.(keys))
    _current_key_idx[] = 1
    println("Registered $(length(_api_keys[])) API key(s)")
    if isempty(_api_keys[])
        println("  *** No valid keys provided! ***")
    end
end

function _get_current_key()
    isempty(_api_keys[]) && error(
        "No API keys registered. Call set_api_keys() first.\n" *
        "Example: set_api_keys([\"gsk_abc123...\", \"gsk_def456...\"])")
    return _api_keys[][_current_key_idx[]]
end

function _rotate_key()
    _current_key_idx[] += 1
    if _current_key_idx[] > length(_api_keys[])
        _current_key_idx[] = 1
        return false  # wrapped around
    end
    println("  Rotating to API key $(_current_key_idx[])/$(length(_api_keys[]))")
    return true
end

# ─────────────────────────────────────────────────────────────────────
# LLM interaction (Groq API with key rotation)
# ─────────────────────────────────────────────────────────────────────

"""
    call_llm(prompt; model=SMALL_MODEL, system="", temperature=0.0) -> String

Send a prompt to the Groq API. Automatically rotates API keys on
rate-limit errors.
"""
function call_llm(prompt::String;
                  model::String=SMALL_MODEL,
                  system::String="",
                  temperature::Float64=0.0)
    messages = Dict{String,String}[]
    if !isempty(system)
        push!(messages, Dict("role" => "system", "content" => system))
    end
    push!(messages, Dict("role" => "user", "content" => prompt))

    body = Dict(
        "model" => model,
        "messages" => messages,
        "stream" => false,
        "temperature" => temperature,
    )

    keys_tried = 0
    t0 = time()

    while keys_tried < length(_api_keys[])
        headers = [
            "Authorization" => "Bearer $(_get_current_key())",
            "Content-Type" => "application/json",
        ]

        resp = HTTP.post(GROQ_API_URL, headers, JSON3.write(body);
                         readtimeout=120, status_exception=false)

        if resp.status == 200
            elapsed = time() - t0
            data = JSON3.read(resp.body)
            content = String(data.choices[1].message.content)
            tokens = haskey(data, :usage) ? data.usage.total_tokens : "?"
            println("  [$(model)] $(length(content)) chars, $(round(elapsed; digits=1))s" *
                    " (key $(_current_key_idx[])/$(length(_api_keys[])), $(tokens) tokens)")
            return content
        elseif resp.status == 429
            retry_after = HTTP.header(resp, "retry-after", "")
            if !isempty(retry_after) && parse(Float64, retry_after) < 30
                wait_secs = parse(Float64, retry_after) + 1
                println("  Rate limited (key $(_current_key_idx[])), waiting $(round(Int, wait_secs))s...")
                sleep(wait_secs)
                continue
            else
                println("  Rate limited (key $(_current_key_idx[])), trying next key...")
            end
        else
            error("Groq API error: $(resp.status)\n$(String(resp.body))")
        end

        keys_tried += 1
        _rotate_key() || break
    end

    error("All API keys rate-limited. Wait a few minutes or add more keys.")
end

"""
    ask_llm_probability(vignette, disease; model=SMALL_MODEL) -> Float64

Ask the LLM to estimate the probability of a disease given a clinical
vignette. Returns a number between 0 and 1. Returns -1.0 if unparseable.
"""
function ask_llm_probability(vignette::String, disease::String;
                             model::String=SMALL_MODEL)
    system = """You are a clinical reasoning assistant. When asked about the
probability of a diagnosis, respond with ONLY a single number between 0.0
and 1.0 representing your estimated probability. No other text."""

    prompt = """Based on the following clinical findings, what is the
probability that the patient has $(disease)?

Clinical findings:
$(vignette)

Respond with a single number between 0.0 and 1.0:"""

    try
        resp = call_llm(prompt; model=model, system=system, temperature=0.0)
        m = match(r"(0?\.\d+|1\.0|0|1)", strip(resp))
        if m !== nothing
            return parse(Float64, m.match)
        end
        @warn "Could not parse LLM probability from: $(resp)"
        return -1.0
    catch e
        # Live API unavailable (no keys, rate-limited, network error).
        # Fall back to the precomputed responses if we have one for this
        # (vignette, disease) pair. If not, propagate the original error.
        precomp = _try_precomputed(vignette, disease)
        if precomp !== nothing
            println("  [precomputed fallback] P($(disease)) = $(precomp)")
            return precomp
        end
        rethrow(e)
    end
end

# ─────────────────────────────────────────────────────────────────────
# Pre-computed LLM fallback
# ─────────────────────────────────────────────────────────────────────

"""
    load_precomputed_llm(filename) -> Dict

Load pre-computed LLM responses from a JSON file. Used as fallback
when the API is unavailable (no wifi, rate limits exhausted).
"""
function load_precomputed_llm(filename::String="llm_precomputed.json")
    path = joinpath(DATA_DIR[], filename)
    if isfile(path)
        data = JSON3.read(read(path, String))
        println("Loaded pre-computed LLM responses from $(filename)")
        return data
    end
    @warn "Pre-computed file not found: $(path)"
    return Dict()
end

# Internal: cached (vignette, disease) -> probability lookup, built lazily
# from llm_precomputed.json + vignettes.json on first need. Engaged
# automatically by ask_llm_probability when the live API call fails.
const _precomputed_cache = Ref{Dict{Tuple{String,String}, Float64}}(
    Dict{Tuple{String,String}, Float64}())
const _precomputed_loaded = Ref(false)

function _build_precomputed_cache!()
    _precomputed_loaded[] = true
    cache = Dict{Tuple{String,String}, Float64}()

    pc_path = joinpath(DATA_DIR[], "llm_precomputed.json")
    if !isfile(pc_path)
        _precomputed_cache[] = cache
        return
    end
    pc = JSON3.read(read(pc_path, String))

    # child_cases: join by id with vignettes.json to recover description text
    vg_path = joinpath(DATA_DIR[], "vignettes.json")
    if isfile(vg_path) && haskey(pc, :child_cases)
        vg = JSON3.read(read(vg_path, String))
        if haskey(vg, :child_cases)
            vg_by_id = Dict(string(c.id) => c for c in vg.child_cases)
            for entry in pc.child_cases
                id = string(entry.id)
                if haskey(vg_by_id, id)
                    desc = String(vg_by_id[id].description)
                    cache[(desc, String(entry.disease))] =
                        Float64(entry.llm_probability)
                end
            end
        end
    end

    _precomputed_cache[] = cache
end

function _try_precomputed(vignette::String, disease::String)
    _precomputed_loaded[] || _build_precomputed_cache!()

    # Exact match on (description, disease) for child_cases vignettes
    cache = _precomputed_cache[]
    haskey(cache, (vignette, disease)) && return cache[(vignette, disease)]

    # The notebook hardcodes its own variants of the stress-test prompts
    # (different wording from vignettes.json's stress_test descriptions),
    # so substring-match those onto the precomputed stress_test entries.
    pc_path = joinpath(DATA_DIR[], "llm_precomputed.json")
    if isfile(pc_path) && disease == "TGA"
        pc = JSON3.read(read(pc_path, String))
        if haskey(pc, :stress_test)
            stress = pc.stress_test
            if occursin("WITH birth asphyxia", vignette) &&
               haskey(stress, :base_rate_shifted)
                return Float64(stress.base_rate_shifted)
            elseif occursin("no birth asphyxia", vignette) &&
                   haskey(stress, :base_rate_original)
                return Float64(stress.base_rate_original)
            elseif occursin("grunting only", vignette) &&
                   haskey(stress, :missing_evidence)
                return Float64(stress.missing_evidence)
            end
        end
    end

    return nothing
end

# ─────────────────────────────────────────────────────────────────────
# Bayesian network helpers
# ─────────────────────────────────────────────────────────────────────

"""
    build_asia_network() -> DiscreteBayesNet

Build the ASIA (Lauritzen-Spiegelhalter 1988) Bayesian network.
INSTRUCTOR version with all CPTs filled in.
"""
function build_asia_network()
    bn = DiscreteBayesNet()

    push!(bn, DiscreteCPD(:Smoker, [0.5, 0.5]))
    push!(bn, DiscreteCPD(:Asia, [0.99, 0.01]))

    push!(bn, CategoricalCPD(:LungCancer, [:Smoker], [2],
        [BNCategorical([0.99, 0.01]),
         BNCategorical([0.90, 0.10])]))

    push!(bn, CategoricalCPD(:Bronchitis, [:Smoker], [2],
        [BNCategorical([0.70, 0.30]),
         BNCategorical([0.40, 0.60])]))

    push!(bn, CategoricalCPD(:Tuberculosis, [:Asia], [2],
        [BNCategorical([0.99, 0.01]),
         BNCategorical([0.95, 0.05])]))

    push!(bn, DiscreteCPD(:TbOrCancer, [:Tuberculosis, :LungCancer], [2, 2],
        [BNCategorical([1.0, 0.0]),
         BNCategorical([0.0, 1.0]),
         BNCategorical([0.0, 1.0]),
         BNCategorical([0.0, 1.0])]))

    push!(bn, CategoricalCPD(:XRay, [:TbOrCancer], [2],
        [BNCategorical([0.95, 0.05]),
         BNCategorical([0.02, 0.98])]))

    push!(bn, CategoricalCPD(:Dyspnoea, [:TbOrCancer, :Bronchitis], [2, 2],
        [BNCategorical([0.90, 0.10]),
         BNCategorical([0.30, 0.70]),
         BNCategorical([0.20, 0.80]),
         BNCategorical([0.10, 0.90])]))

    return bn
end

"""
    query_bn(bn, target; evidence=Assignment()) -> Vector{Float64}

Query a Bayesian network for the posterior distribution of `target`
given `evidence`. Returns a probability vector.
"""
function query_bn(bn::DiscreteBayesNet, target::Symbol;
                  evidence::Assignment=Assignment())
    result = infer(bn, target; evidence=evidence)
    return result.potential
end

function query_bn(bn::DiscreteBayesNet, target::Symbol,
                  evidence::Dict)
    return query_bn(bn, target; evidence=Assignment(evidence))
end

function query_bn_all(bn::DiscreteBayesNet, targets::Vector{Symbol};
                      evidence::Assignment=Assignment())
    return Dict(t => query_bn(bn, t; evidence=evidence) for t in targets)
end

bn_node_names(bn::DiscreteBayesNet) = names(bn)
bn_parents(bn::DiscreteBayesNet, node::Symbol) = parents(bn, node)

# ─────────────────────────────────────────────────────────────────────
# Scoring
# ─────────────────────────────────────────────────────────────────────

function brier_score(predictions::Vector{Float64}, outcomes::Vector{Int})
    @assert length(predictions) == length(outcomes)
    return sum((predictions .- outcomes).^2) / length(predictions)
end

function calibration_error(predictions::Vector{Float64}, outcomes::Vector{Int})
    @assert length(predictions) == length(outcomes)
    return sum(abs.(predictions .- outcomes)) / length(predictions)
end

function score_round(estimates::Vector{Float64}, ground_truth::Vector{Int})
    bs = brier_score(estimates, ground_truth)
    ce = calibration_error(estimates, ground_truth)
    composite = round((1 - bs) * 100; digits=1)
    return Dict(
        "composite" => composite,
        "brier_score" => round(bs; digits=4),
        "calibration_error" => round(ce; digits=4),
        "n_cases" => length(estimates),
    )
end

function show_calibration(label::String,
                          bn_est::Vector{Float64},
                          llm_est::Vector{Float64},
                          human_est::Vector{Float64},
                          truth::Vector{Int})
    bn_score = score_round(bn_est, truth)
    llm_score = score_round(llm_est, truth)
    human_score = score_round(human_est, truth)

    println("\n=== $(label) ===\n")
    println("  Source          Brier    Calibration   Composite")
    println("  " * "-"^52)
    for (name, s) in [("Bayesian Net", bn_score),
                       ("LLM", llm_score),
                       ("Your Intuition", human_score)]
        line = @sprintf("  %-16s %6.4f   %11.4f   %9.1f",
                        name, s["brier_score"], s["calibration_error"], s["composite"])
        println(line)
    end
end

# ─────────────────────────────────────────────────────────────────────
# Leaderboard
# ─────────────────────────────────────────────────────────────────────

function submit_to_leaderboard(round_num::Int, result::Dict)
    if occursin("REPLACE", FORM_URL[])
        println("[Leaderboard not configured] Round $(round_num): composite=$(result["composite"])")
        return
    end
    if GROUP_NAME[] == "CHANGE_ME"
        println("[GROUP_NAME not set] Submission would tag your team as 'CHANGE_ME' — skipping. Set LabUtils.GROUP_NAME[] = \"YourTeam\" first.")
        return
    end
    # Google Forms accepts both multipart and URL-encoded bodies, but
    # multipart submissions are sometimes silently dropped. URL-encoded
    # is the reliable path.
    body = string(
        FIELD_GROUP[],  "=", HTTP.escapeuri(GROUP_NAME[]),
        "&", FIELD_ROUND[],  "=", round_num,
        "&", FIELD_SCORE[],  "=", result["composite"],
        "&", FIELD_DETAIL[], "=", HTTP.escapeuri(JSON3.write(result)),
    )
    try
        resp = HTTP.post(FORM_URL[],
                         ["Content-Type" => "application/x-www-form-urlencoded"],
                         body;
                         status_exception=false, readtimeout=10)
        if resp.status == 200
            println("✓ Submitted Round $(round_num): composite=$(result["composite"])")
        else
            println("✗ Submission HTTP $(resp.status) for Round $(round_num)")
        end
    catch e
        println("Submission failed: $(e)")
    end
end

# ─────────────────────────────────────────────────────────────────────
# Verification
# ─────────────────────────────────────────────────────────────────────

function verify_setup()
    println("Checking data files...")
    for f in ["vignettes.json", "llm_precomputed.json"]
        path = joinpath(DATA_DIR[], f)
        status = isfile(path) ? "found" : (f == "llm_precomputed.json" ? "not found (optional)" : "NOT FOUND")
        println("  $(f): $(status)")
    end

    println("\nChecking Groq API...")
    if isempty(_api_keys[])
        println("  No API keys registered yet.")
        println("  *** Call set_api_keys([\"gsk_...\", \"gsk_...\"]) ***")
    else
        try
            resp = HTTP.get("https://api.groq.com/openai/v1/models",
                           headers=["Authorization" => "Bearer $(_get_current_key())"],
                           readtimeout=10, status_exception=false)
            if resp.status == 200
                println("  Groq API: connected ✓")
                println("  $(length(_api_keys[])) API key(s) registered")
            else
                println("  Groq API: error (status $(resp.status))")
            end
        catch e
            println("  Groq API not reachable: $(e)")
            println("  Pre-computed LLM responses will be used as fallback.")
        end
    end

    println("\nGroup: $(GROUP_NAME[])")
    if GROUP_NAME[] == "CHANGE_ME"
        println("  *** Set your group name in the notebook ***")
    end

    println("\nBayesNets.jl: loaded ✓")
end

# ─────────────────────────────────────────────────────────────────────
# CHILD Bayesian Network (Spiegelhalter & Cowell, 1992)
# 20 nodes, 230+ parameters — congenital heart disease diagnosis
# Pre-built from child.bif; no file parsing needed at runtime.
# ─────────────────────────────────────────────────────────────────────

function build_child_network()
    bn = DiscreteBayesNet()

    push!(bn, DiscreteCPD(:BirthAsphyxia, [0.10000000, 0.90000000]))
    push!(bn, CategoricalCPD(:Disease, [:BirthAsphyxia], [2],
        [
         BNCategorical([0.20000000, 0.30000000, 0.25000000, 0.15000000, 0.05000000, 0.05000000]),
         BNCategorical([0.03061224, 0.33673469, 0.29591837, 0.23469388, 0.05102041, 0.05102041])
        ]))
    push!(bn, CategoricalCPD(:DuctFlow, [:Disease], [6],
        [
         BNCategorical([0.15000000, 0.05000000, 0.80000000]),
         BNCategorical([0.10000000, 0.80000000, 0.10000000]),
         BNCategorical([0.80000000, 0.20000000, 0.00000000]),
         BNCategorical([1.00000000, 0.00000000, 0.00000000]),
         BNCategorical([0.33000000, 0.33000000, 0.34000000]),
         BNCategorical([0.20000000, 0.40000000, 0.40000000])
        ]))
    push!(bn, CategoricalCPD(:CardiacMixing, [:Disease], [6],
        [
         BNCategorical([0.40000000, 0.43000000, 0.15000000, 0.02000000]),
         BNCategorical([0.02000000, 0.09000000, 0.09000000, 0.80000000]),
         BNCategorical([0.02000000, 0.16000000, 0.80000000, 0.02000000]),
         BNCategorical([0.01000000, 0.02000000, 0.95000000, 0.02000000]),
         BNCategorical([0.01000000, 0.03000000, 0.95000000, 0.01000000]),
         BNCategorical([0.40000000, 0.53000000, 0.05000000, 0.02000000])
        ]))
    push!(bn, CategoricalCPD(:HypDistrib, [:DuctFlow, :CardiacMixing], [3, 4],
        [
         BNCategorical([0.95000000, 0.05000000]),
         BNCategorical([0.95000000, 0.05000000]),
         BNCategorical([0.05000000, 0.95000000]),
         BNCategorical([0.95000000, 0.05000000]),
         BNCategorical([0.95000000, 0.05000000]),
         BNCategorical([0.50000000, 0.50000000]),
         BNCategorical([0.95000000, 0.05000000]),
         BNCategorical([0.95000000, 0.05000000]),
         BNCategorical([0.95000000, 0.05000000]),
         BNCategorical([0.95000000, 0.05000000]),
         BNCategorical([0.95000000, 0.05000000]),
         BNCategorical([0.50000000, 0.50000000])
        ]))
    push!(bn, CategoricalCPD(:LungParench, [:Disease], [6],
        [
         BNCategorical([0.60000000, 0.10000000, 0.30000000]),
         BNCategorical([0.80000000, 0.05000000, 0.15000000]),
         BNCategorical([0.80000000, 0.05000000, 0.15000000]),
         BNCategorical([0.80000000, 0.05000000, 0.15000000]),
         BNCategorical([0.10000000, 0.60000000, 0.30000000]),
         BNCategorical([0.03000000, 0.25000000, 0.72000000])
        ]))
    push!(bn, CategoricalCPD(:HypoxiaInO2, [:CardiacMixing, :LungParench], [4, 3],
        [
         BNCategorical([0.93000000, 0.05000000, 0.02000000]),
         BNCategorical([0.10000000, 0.80000000, 0.10000000]),
         BNCategorical([0.10000000, 0.70000000, 0.20000000]),
         BNCategorical([0.02000000, 0.18000000, 0.80000000]),
         BNCategorical([0.15000000, 0.80000000, 0.05000000]),
         BNCategorical([0.10000000, 0.75000000, 0.15000000]),
         BNCategorical([0.05000000, 0.65000000, 0.30000000]),
         BNCategorical([0.10000000, 0.30000000, 0.60000000]),
         BNCategorical([0.70000000, 0.20000000, 0.10000000]),
         BNCategorical([0.10000000, 0.65000000, 0.25000000]),
         BNCategorical([0.10000000, 0.50000000, 0.40000000]),
         BNCategorical([0.02000000, 0.18000000, 0.80000000])
        ]))
    push!(bn, CategoricalCPD(:LowerBodyO2, [:HypDistrib, :HypoxiaInO2], [2, 3],
        [
         BNCategorical([0.10000000, 0.30000000, 0.60000000]),
         BNCategorical([0.40000000, 0.50000000, 0.10000000]),
         BNCategorical([0.30000000, 0.60000000, 0.10000000]),
         BNCategorical([0.50000000, 0.45000000, 0.05000000]),
         BNCategorical([0.50000000, 0.40000000, 0.10000000]),
         BNCategorical([0.60000000, 0.35000000, 0.05000000])
        ]))
    push!(bn, CategoricalCPD(:RUQO2, [:HypoxiaInO2], [3],
        [
         BNCategorical([0.10000000, 0.30000000, 0.60000000]),
         BNCategorical([0.30000000, 0.60000000, 0.10000000]),
         BNCategorical([0.50000000, 0.40000000, 0.10000000])
        ]))
    push!(bn, CategoricalCPD(:CO2, [:LungParench], [3],
        [
         BNCategorical([0.80000000, 0.10000000, 0.10000000]),
         BNCategorical([0.65000000, 0.05000000, 0.30000000]),
         BNCategorical([0.45000000, 0.05000000, 0.50000000])
        ]))
    push!(bn, CategoricalCPD(:CO2Report, [:CO2], [3],
        [
         BNCategorical([0.90000000, 0.10000000]),
         BNCategorical([0.90000000, 0.10000000]),
         BNCategorical([0.10000000, 0.90000000])
        ]))
    push!(bn, CategoricalCPD(:LungFlow, [:Disease], [6],
        [
         BNCategorical([0.30000000, 0.65000000, 0.05000000]),
         BNCategorical([0.20000000, 0.05000000, 0.75000000]),
         BNCategorical([0.15000000, 0.80000000, 0.05000000]),
         BNCategorical([0.10000000, 0.85000000, 0.05000000]),
         BNCategorical([0.30000000, 0.10000000, 0.60000000]),
         BNCategorical([0.70000000, 0.10000000, 0.20000000])
        ]))
    push!(bn, CategoricalCPD(:ChestXray, [:LungParench, :LungFlow], [3, 3],
        [
         BNCategorical([0.90000000, 0.03000000, 0.03000000, 0.01000000, 0.03000000]),
         BNCategorical([0.05000000, 0.02000000, 0.15000000, 0.70000000, 0.08000000]),
         BNCategorical([0.05000000, 0.05000000, 0.05000000, 0.05000000, 0.80000000]),
         BNCategorical([0.14000000, 0.80000000, 0.02000000, 0.02000000, 0.02000000]),
         BNCategorical([0.05000000, 0.22000000, 0.08000000, 0.50000000, 0.15000000]),
         BNCategorical([0.05000000, 0.15000000, 0.05000000, 0.05000000, 0.70000000]),
         BNCategorical([0.15000000, 0.01000000, 0.79000000, 0.04000000, 0.01000000]),
         BNCategorical([0.05000000, 0.02000000, 0.40000000, 0.40000000, 0.13000000]),
         BNCategorical([0.24000000, 0.33000000, 0.03000000, 0.34000000, 0.06000000])
        ]))
    push!(bn, CategoricalCPD(:XrayReport, [:ChestXray], [5],
        [
         BNCategorical([0.80000000, 0.06000000, 0.06000000, 0.02000000, 0.06000000]),
         BNCategorical([0.10000000, 0.80000000, 0.02000000, 0.02000000, 0.06000000]),
         BNCategorical([0.10000000, 0.02000000, 0.80000000, 0.02000000, 0.06000000]),
         BNCategorical([0.08000000, 0.02000000, 0.10000000, 0.60000000, 0.20000000]),
         BNCategorical([0.08000000, 0.02000000, 0.10000000, 0.10000000, 0.70000000])
        ]))
    push!(bn, CategoricalCPD(:Sick, [:Disease], [6],
        [
         BNCategorical([0.40000000, 0.60000000]),
         BNCategorical([0.30000000, 0.70000000]),
         BNCategorical([0.20000000, 0.80000000]),
         BNCategorical([0.30000000, 0.70000000]),
         BNCategorical([0.70000000, 0.30000000]),
         BNCategorical([0.70000000, 0.30000000])
        ]))
    push!(bn, CategoricalCPD(:Grunting, [:LungParench, :Sick], [3, 2],
        [
         BNCategorical([0.20000000, 0.80000000]),
         BNCategorical([0.40000000, 0.60000000]),
         BNCategorical([0.80000000, 0.20000000]),
         BNCategorical([0.05000000, 0.95000000]),
         BNCategorical([0.20000000, 0.80000000]),
         BNCategorical([0.60000000, 0.40000000])
        ]))
    push!(bn, CategoricalCPD(:GruntingReport, [:Grunting], [2],
        [
         BNCategorical([0.80000000, 0.20000000]),
         BNCategorical([0.10000000, 0.90000000])
        ]))
    push!(bn, CategoricalCPD(:LVH, [:Disease], [6],
        [
         BNCategorical([0.10000000, 0.90000000]),
         BNCategorical([0.10000000, 0.90000000]),
         BNCategorical([0.10000000, 0.90000000]),
         BNCategorical([0.90000000, 0.10000000]),
         BNCategorical([0.05000000, 0.95000000]),
         BNCategorical([0.10000000, 0.90000000])
        ]))
    push!(bn, CategoricalCPD(:LVHreport, [:LVH], [2],
        [
         BNCategorical([0.90000000, 0.10000000]),
         BNCategorical([0.05000000, 0.95000000])
        ]))
    push!(bn, CategoricalCPD(:Age, [:Disease, :Sick], [6, 2],
        [
         BNCategorical([0.95000000, 0.03000000, 0.02000000]),
         BNCategorical([0.80000000, 0.15000000, 0.05000000]),
         BNCategorical([0.70000000, 0.15000000, 0.15000000]),
         BNCategorical([0.80000000, 0.15000000, 0.05000000]),
         BNCategorical([0.80000000, 0.15000000, 0.05000000]),
         BNCategorical([0.90000000, 0.08000000, 0.02000000]),
         BNCategorical([0.85000000, 0.10000000, 0.05000000]),
         BNCategorical([0.70000000, 0.20000000, 0.10000000]),
         BNCategorical([0.25000000, 0.25000000, 0.50000000]),
         BNCategorical([0.80000000, 0.15000000, 0.05000000]),
         BNCategorical([0.70000000, 0.20000000, 0.10000000]),
         BNCategorical([0.80000000, 0.15000000, 0.05000000])
        ]))

    return bn
end

end # module
